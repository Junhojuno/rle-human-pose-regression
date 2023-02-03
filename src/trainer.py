import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.losses import RLELoss
from src.metrics import calc_coord_accuracy
from src.scheduler import MultiStepLR


class Trainer:
    """Train class"""

    def __init__(self, args, model, logger, strategy) -> None:
        self.args = args
        self.epochs = args.TRAIN.EPOCHS
        self.n_train_steps = int(
            args.DATASET.TRAIN.EXAMPLES
            // (args.TRAIN.BATCH_SIZE * strategy.num_replicas_in_sync)
        )
        os.makedirs(self.args.OUTPUT.CKPT, exist_ok=True)

        self.checkpoint_prefix = os.path.join(
            args.OUTPUT.CKPT, "best_model.tf"
        )

        self.model = model
        self.loss_object = RLELoss()
        lr_scheduler = MultiStepLR(
            args.TRAIN.LR,
            lr_steps=[
                self.n_train_steps * epoch for epoch in args.TRAIN.LR_EPOCHS
            ],
            lr_rate=args.TRAIN.LR_FACTOR
        )
        self.optimizer = Adam(learning_rate=lr_scheduler)

        self.logger = logger
        self.strategy = strategy

    def train_step(self, inputs):
        images, targets = inputs

        with tf.GradientTape() as tape:
            pred = self.model(images, mu_g=targets[..., :2], training=True)
            loss = self.loss_object(targets, pred)
            loss = tf.math.reduce_mean(loss) \
                * (1. / self.strategy.num_replicas_in_sync)
            loss += sum(self.model.losses) \
                * (1. / self.strategy.num_replicas_in_sync)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        acc = calc_coord_accuracy(
            targets, pred, self.args.DATASET.COMMON.OUTPUT_SHAPE
        )
        acc *= (1. / self.strategy.num_replicas_in_sync)
        return loss, acc

    def val_step(self, inputs):
        images, targets = inputs
        pred = self.model(images, mu_g=targets[..., :2], training=False)
        loss = self.loss_object(targets, pred)
        loss = tf.math.reduce_mean(loss)
        loss *= (1. / self.strategy.num_replicas_in_sync)
        acc = calc_coord_accuracy(
            targets, pred, self.args.DATASET.COMMON.OUTPUT_SHAPE
        )
        acc *= (1. / self.strategy.num_replicas_in_sync)
        return loss, acc

    @tf.function
    def distributed_train_step(self, inputs):
        per_replica_loss, per_acc = self.strategy.run(
            self.train_step, args=(inputs,)
        )
        step_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
        )
        step_acc = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_acc, axis=None
        )
        return step_loss, step_acc

    @tf.function
    def distributed_val_step(self, inputs):
        per_replica_loss, per_acc = self.strategy.run(
            self.val_step, args=(inputs,)
        )
        step_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
        )
        step_acc = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_acc, axis=None
        )
        return step_loss, step_acc

    def custom_loop(self, train_dataset, val_dataset, wandb_run=None):
        # best_acc = 1e-10
        lowest_loss = 1e+15
        for epoch in tf.range(self.args.TRAIN.EPOCHS, dtype=tf.int64):
            train_loss, val_loss = 0.0, 0.0
            train_acc, val_acc = 0.0, 0.0
            train_n_batches, val_n_batches = 0.0, 0.0

            start_time = time.time()
            for inputs in train_dataset:
                loss, acc = self.distributed_train_step(inputs)
                train_loss += loss
                train_acc += acc
                train_n_batches += 1
            train_acc = train_acc / train_n_batches
            train_loss = train_loss / train_n_batches
            train_time = time.time() - start_time

            for inputs in val_dataset:
                loss, acc = self.distributed_val_step(inputs)
                val_loss += loss
                val_acc += acc
                val_n_batches += 1
            val_acc = val_acc / val_n_batches
            val_loss = val_loss / val_n_batches
            total_time = time.time() - start_time

            # current_lr = self.optimizer.lr(self.optimizer.iterations).numpy()
            current_lr = self.optimizer.lr.numpy()

            self.logger.info(
                'Epoch: {:03d} - {}s[{}s] | Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | LR: {}'.format(
                    epoch + 1,
                    int(total_time),
                    int(train_time),
                    float(train_loss),
                    float(train_acc),
                    float(val_loss),
                    float(val_acc),
                    float(current_lr)
                )
            )
            if wandb_run:
                # write on wandb server
                wandb_run.log(
                    {
                        'loss/train': float(train_loss),
                        'loss/val': float(val_loss),
                        'acc/train': float(train_acc),
                        'acc/val': float(val_acc),
                        'lr': float(current_lr),
                        'epoch': int(epoch + 1)
                    }
                )
            # Terminate when NaN loss
            if tf.math.is_nan(train_loss) or tf.math.is_nan(val_loss):
                self.logger.info('Training is Terminated because of NaN Loss.')
                raise ValueError('NaN Loss has coming up.')

            # save model weights
            self.model.save_weights(
                self.checkpoint_prefix.replace('best', 'newest')
            )
            curr_val_loss = float(val_loss)
            if curr_val_loss < lowest_loss:
                lowest_loss = curr_val_loss
                self.model.save_weights(self.checkpoint_prefix)
