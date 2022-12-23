import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts
from losses import RLELoss
from metrics import calc_coord_accuracy
from scheduler import MultiStepLR


class Trainer:
    """Train class"""
  
    def __init__(self, args, model, logger, strategy) -> None:
        self.args = args
        self.epochs = args.train.n_epochs
        self.n_train_steps = int(
            args.dataset.n_train_examples // (args.train.batch_size * strategy.num_replicas_in_sync)
        )
        try:
            os.mkdir(self.args.ckpt_dir)
        except FileExistsError:
            pass
        
        self.checkpoint_prefix = os.path.join(self.args.ckpt_dir, "best_model.tf")
                
        self.model = model
        self.loss_object = RLELoss()
        lr_scheduler = MultiStepLR(
            args.train.lr,
            lr_steps=[self.n_train_steps * epoch for epoch in args.train.scheduler.lr_epochs],
            lr_rate=args.train.scheduler.lr_rate
        )
        self.optimizer = Adam(learning_rate=lr_scheduler)
        
        self.logger = logger
        self.strategy = strategy
        
        self.model.summary(print_fn=self.logger.info)
        
        self.logger.info(f'==== Backbone: ResNet50')
        self.logger.info(f'==== Input : {self.args.dataset.input_shape[0]}x{self.args.dataset.input_shape[1]}')
        self.logger.info(f'==== Batch size: {args.train.batch_size * strategy.num_replicas_in_sync}')
        self.logger.info(f'==== Dataset: {self.args.dataset.name}')
    
    def train_step(self, inputs):
        images, targets = inputs
        
        with tf.GradientTape() as tape:
            pred = self.model(images, mu_g=targets[..., :2], training=True)  # pred: EasyDict
            loss = self.loss_object(targets, pred)
            loss = tf.math.reduce_mean(loss) * (1. / self.strategy.num_replicas_in_sync)
            loss += (sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        acc = calc_coord_accuracy(targets, pred, self.args.dataset.output_shape) * (1. / self.strategy.num_replicas_in_sync)
        return loss, acc

    def val_step(self, inputs):
        images, targets = inputs

        pred = self.model(images, mu_g=targets[..., :2], training=False)  # pred: EasyDict
        loss = self.loss_object(targets, pred)
        loss = tf.math.reduce_mean(loss) * (1. / self.strategy.num_replicas_in_sync)
        acc = calc_coord_accuracy(targets, pred, self.args.dataset.output_shape) * (1. / self.strategy.num_replicas_in_sync)
        return loss, acc
    
    @tf.function
    def distributed_train_step(self, inputs):
        per_replica_loss, per_acc = self.strategy.run(self.train_step, args=(inputs,))
        step_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        step_acc = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_acc, axis=None)
        return step_loss, step_acc
    
    @tf.function
    def distributed_val_step(self, inputs):
        per_replica_loss, per_acc = self.strategy.run(self.val_step, args=(inputs,))
        step_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        step_acc = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_acc, axis=None)
        return step_loss, step_acc
    
    def custom_loop(self, train_dataset, val_dataset, wandb_run=None):
        # best_acc = 1e-10
        lowest_loss = 1e+15
        start = self.args.train.start_epoch_index
        end = self.args.train.start_epoch_index + self.args.train.n_epochs
        
        for epoch in tf.range(start, end, dtype=tf.int64):
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
            
            if self.args.train.scheduler.name != 'no_scheduler':
                current_lr = self.optimizer.lr(self.optimizer.iterations).numpy()
            else:
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

            # Terminate when NaN loss
            if tf.math.is_nan(train_loss) or tf.math.is_nan(val_loss):
                self.logger.info('Training  is Terminated because of NaN Loss.')
                raise ValueError('NaN Loss has coming up.')
            
            # save model weights
            self.model.save_weights(self.checkpoint_prefix.replace('best', 'newest'))

            curr_val_loss = float(val_loss)
            if curr_val_loss < lowest_loss:
                lowest_loss = curr_val_loss
                self.model.save_weights(self.checkpoint_prefix)
