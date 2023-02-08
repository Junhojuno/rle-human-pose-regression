"""Custom training pipeline"""
from pathlib import Path
import os
import argparse
import time
import tensorflow as tf

import wandb

from src.model import RLEModel
from src.function import train, validate
from src.losses import RLELoss
from src.scheduler import MultiStepLR
from src.dataset import load_dataset
from src.evaluate import load_eval_dataset, STATS_NAMES
from src.utils import (
    parse_yaml,
    get_flops,
    get_logger
)


def define_argparser():
    parser = argparse.ArgumentParser('human pose regression parser')
    parser.add_argument(
        '--config', '-c',
        dest='config',
        required=True,
        help='yaml file path'
    )
    return parser.parse_args()


def main():
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)

    cfg = define_argparser()
    args = parse_yaml(cfg.config)

    args.DATASET.COMMON.OUTPUT_SHAPE = [
        args.DATASET.COMMON.INPUT_SHAPE[0] // 4,
        args.DATASET.COMMON.INPUT_SHAPE[1] // 4
    ]

    cwd = Path('./').resolve()
    args.WANDB.NAME = \
        '{dataset}/{exp_title}/{model}/'\
        '{backbone}/bs{bs}_lr{lr}_s{sigma}_sf_{sf}_r{rot}'\
        .format(
            dataset=args.DATASET.NAME,
            exp_title=args.WANDB.SUBTITLE,
            model=args.MODEL.NAME,
            backbone=args.MODEL.BACKBONE,
            bs=args.TRAIN.BATCH_SIZE,
            lr=args.TRAIN.LR,
            sigma=args.DATASET.COMMON.SIGMA,
            sf=args.AUG.SCALE_FACTOR,
            rot=args.AUG.ROT_FACTOR
        )
    args.OUTPUT.DIR = f'results/{args.WANDB.NAME}'

    # set save_ckpt dir
    args.OUTPUT.CKPT = f'{args.OUTPUT.DIR}/ckpt'
    os.makedirs(args.OUTPUT.DIR, exist_ok=True)

    logger = get_logger(f'{args.OUTPUT.DIR}/work.log')

    train_ds = load_dataset(
        str(
            cwd.parent
            / 'datasets'
            / args.DATASET.NAME
            / args.DATASET.TRAIN.PATTERN
        ),
        args.TRAIN.BATCH_SIZE,
        args,
        'train',
        use_aug=True
    )
    eval_ds = None
    if args.EVAL.DO_EVAL:
        eval_ds = load_eval_dataset(
            str(
                cwd.parent
                / 'datasets'
                / args.DATASET.NAME
                / args.DATASET.VAL.PATTERN
            ),
            args.EVAL.BATCH_SIZE,
            args.DATASET.COMMON.K,
            args.DATASET.COMMON.INPUT_SHAPE
        )
    # initialize wandb
    run = None
    if args.WANDB.USE:
        run = wandb.init(
            project=args.WANDB.PROJECT,
            config=args,
            name=args.WANDB.NAME,
            resume=False,
            id=None,
            dir=args.OUTPUT.DIR  # should be generated right before
        )
        # define our custom x axis metric
        run.define_metric("epoch")
        run.define_metric("eval")
        # define which metrics will be plotted against it
        run.define_metric("loss/*", step_metric="epoch")
        run.define_metric("acc/*", step_metric="epoch")
        run.define_metric("lr", step_metric="epoch")

    model = RLEModel(
        args.DATASET.COMMON.K,
        args.DATASET.COMMON.INPUT_SHAPE,
        args.MODEL.BACKBONE,
        is_training=True
    )
    model.build([None, *args.DATASET.COMMON.INPUT_SHAPE])
    model.summary(print_fn=logger.info)
    flops = get_flops(model, args.DATASET.COMMON.INPUT_SHAPE)
    logger.info(
        f'===={args.MODEL.NAME}====\n'
        f'==== Backbone: {args.MODEL.BACKBONE}\n'
        f'==== Input : {args.DATASET.COMMON.INPUT_SHAPE}\n'
        f'==== Batch size: {args.TRAIN.BATCH_SIZE}\n'
        f'==== Dataset: {args.DATASET.NAME}\n'
        f'==== GFLOPs: {flops}'
    )
    n_train_steps = int(
        args.DATASET.TRAIN.EXAMPLES // args.TRAIN.BATCH_SIZE
    )
    os.makedirs(args.OUTPUT.CKPT, exist_ok=True)

    checkpoint_prefix = os.path.join(
        args.OUTPUT.CKPT, "best_model.tf"
    )
    criterion = RLELoss()
    lr_scheduler = MultiStepLR(
        args.TRAIN.LR,
        lr_steps=[
            n_train_steps * epoch
            for epoch in args.TRAIN.LR_EPOCHS
        ],
        lr_rate=args.TRAIN.LR_FACTOR
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    best_ap = 1e-5
    for epoch in tf.range(args.TRAIN.EPOCHS, dtype=tf.int64):
        train_loss, train_acc, train_n_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for inputs in train_ds:
            loss, acc = train(inputs, model, criterion, optimizer, args)
            train_loss += loss
            train_acc += acc
            train_n_batches += 1
        train_acc = train_acc / train_n_batches
        train_loss = train_loss / train_n_batches
        train_time = time.time() - start_time
        current_lr = optimizer.lr.numpy()
        logger.info(
            f'Epoch: {epoch + 1:03d} | [{int(train_time)}s] '
            f'| Train Loss: {float(train_loss):.4f}'
            f'| Train Acc: {float(train_acc):.4f}'
            f'| LR: {float(current_lr)}'
        )
        if run:  # write on wandb server
            run.log(
                {
                    'loss/train': float(train_loss),
                    'acc/train': float(train_acc),
                    'lr': float(current_lr),
                    'epoch': int(epoch + 1)
                }
            )
        # Terminate when NaN loss
        if tf.math.is_nan(train_loss):
            logger.info('Training is Terminated because of NaN Loss.')
            raise ValueError('NaN Loss has coming up.')

        # save model newest weights
        model.save_weights(
            checkpoint_prefix.replace('best', 'newest')
        )
        if (eval_ds is not None)\
                and (epoch + 1) % args.EVAL.EVAL_TERM == 0:
            assert args.EVAL.DO_EVAL,\
                'evaluation must be done.'\
                f'but, received DO_EVAL: {args.EVAL.DO_EVAL}'
            aps = validate(
                model,
                eval_ds,
                args.DATASET.COMMON.INPUT_SHAPE,
                str(cwd.parent / args.EVAL.COCO_JSON),
                args.EVAL.FLIP_TEST
            )
            if aps[0] > best_ap:
                best_ap = aps[0]
                model.save_weights(checkpoint_prefix)

    # final evaluation with best model
    model = RLEModel(
        args.DATASET.COMMON.K,
        args.DATASET.COMMON.INPUT_SHAPE,
        args.MODEL.BACKBONE,
        is_training=False
    )
    model.load_weights(checkpoint_prefix)
    aps = validate(
        model,
        eval_ds,
        args.DATASET.COMMON.INPUT_SHAPE,
        str(cwd.parent / args.EVAL.COCO_JSON),
        args.EVAL.FLIP_TEST
    )
    if args.WANDB.USE:
        eval_table = wandb.Table(
            data=[aps],
            columns=STATS_NAMES
        )
        run.log({'eval': eval_table})


if __name__ == '__main__':
    main()
