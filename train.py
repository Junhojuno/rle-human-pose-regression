"""Custom training pipeline"""
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import logging
import argparse
from collections import OrderedDict

import wandb

from src.model import RLEModel
from src.trainer import Trainer
from src.dataset import load_dataset
from src.utils import parse_yaml, get_flops
from evaluate import evaluate_coco, print_name_value

logger = logging.getLogger(__name__)


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
    cfg = define_argparser()
    args = parse_yaml(cfg.config)

    args.DATASET.COMMON.OUTPUT_SHAPE = [
        args.DATASET.COMMON.INPUT_SHAPE[0] // 4,
        args.DATASET.COMMON.INPUT_SHAPE[1] // 4
    ]

    cwd = Path('./').resolve()
    args.WANDB.NAME = \
        '{dataset}/{exp_title}/{model}/\
            {backbone}/bs{bs}_lr{lr}_s{sigma}_sf_{sf}_r{rot}'\
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

    strategy = tf.distribute.MirroredStrategy()

    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)

    train_ds = load_dataset(
        str(cwd.parent / 'datasets' / args.DATASET.NAME / args.DATASET.TRAIN.PATTERN),
        args.TRAIN.BATCH_SIZE * strategy.num_replicas_in_sync,
        args,
        'train',
        use_aug=True
    )
    val_ds = load_dataset(
        str(cwd.parent / 'datasets' / args.DATASET.NAME / args.DATASET.VAL.PATTERN),
        args.VAL.BATCH_SIZE * strategy.num_replicas_in_sync,
        args,
        'val',
        use_aug=False
    )
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

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

    with strategy.scope():
        model = RLEModel(
            args.DATASET.COMMON.K,
            args.DATASET.COMMON.INPUT_SHAPE,
            args.MODEL.BACKBONE,
            is_training=True
        )
        model.build([None, *args.DATASET.COMMON.INPUT_SHAPE])
        # model.summary(print_fn=logger.info)
        flops = get_flops(model, args.DATASET.COMMON.INPUT_SHAPE)
        logger.info(
            f'===={args.MODEL.NAME}====\n'
            f'==== Backbone: {args.MODEL.BACKBONE}'
            f'==== Input : {args.DATASET.COMMON.INPUT_SHAPE}'
            f'==== Batch size: {args.TRAIN.BATCH_SIZE * strategy.num_replicas_in_sync}'
            f'==== Dataset: {args.DATASET.NAME}',
            f'==== GFLOPs: {flops}'
        )
        # train
        trainer = Trainer(args, model, logger, strategy)
        trainer.custom_loop(
            train_dist_ds,
            val_dist_ds,
            run
        )
    # evaluation
    if args.DATASET.VAL.USE_EVAL:
        model = RLEModel(
            args.DATASET.COMMON.K,
            args.DATASET.COMMON.INPUT_SHAPE,
            args.MODEL.BACKBONE,
            is_training=False
        )  # model will return heatmaps
        model.load_weights(trainer.checkpoint_prefix)
        stats = evaluate_coco(
            model,
            str(cwd.parent / args.DATASET.VAL.PATTERN),
            args.DATASET.COMMON.K,
            args.DATASET.COMMON.INPUT_SHAPE,
            str(cwd.parent / args.DATASET.VAL.COCO_JSON)
        )
        stats_names = [
            "AP",
            "Ap .5",
            "AP .75",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (M)",
            "AR (L)",
        ]

        if args.WANDB.USE:
            eval_table = wandb.Table(
                data=[stats],
                columns=stats_names
            )
            run.log({'eval': eval_table})
        else:
            info_str = []
            for i, name in enumerate(stats_names):
                info_str.append((name, stats[i]))

            results = OrderedDict(info_str)
            print_name_value(results)


if __name__ == '__main__':
    main()
