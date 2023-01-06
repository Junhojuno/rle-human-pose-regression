"""Custom training pipeline"""
from pathlib import Path
import numpy as np
import tensorflow as tf
import logging
from collections import OrderedDict

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import wandb

from src.model import RLEModel
from src.trainer import Trainer
from src.dataset import load_dataset
from src.utils import get_available_gpu, to_dict, to_namedtuple
from evaluate import evaluate_coco, print_name_value

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    
    # calculate output shape
    cfg['dataset']['output_shape'] = [
        cfg['dataset']['input_shape'][0] // 4,
        cfg['dataset']['input_shape'][1] // 4
    ]  # calculate output shape

    # set save_ckpt dir
    cfg['ckpt_dir'] = '{cwd}/results/{name}/ckpt'.format(
        cwd=cfg['original_work_dir'],
        name=cfg['wandb']['name']
    )
    if cfg['additional']['use_qat']:
        cfg['ckpt_dir'] += '_qat'
    args = to_namedtuple(cfg)

    gpus = get_available_gpu()
    n_usable_gpus = len(gpus)
    if args.train.n_gpus > n_usable_gpus:
        raise ValueError(
            f'train.n_gpus is not avaliable. Only {n_usable_gpus} usable, but given {args.train.n_gpus}'
        )
    strategy = tf.distribute.MirroredStrategy(gpus[:args.train.n_gpus])

    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)

    cwd = Path(get_original_cwd()).resolve()
    train_ds = load_dataset(
        str(cwd.parent / args.dataset.train_pattern),
        args.train.batch_size * strategy.num_replicas_in_sync,
        args,
        'train',
        use_aug=True
    )
    val_ds = load_dataset(
        str(cwd.parent / args.dataset.val_pattern),
        args.val.batch_size * strategy.num_replicas_in_sync,
        args,
        'val',
        use_aug=False
    )
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

    weight_path = None
    if args.train.use_pretrained:
        weight_path = str(cwd / args.train.pretrained_weights)

    # initialize wandb
    run = None
    if args.wandb.use:
        if args.wandb.resume:
            assert args.wandb.run_id is not None,\
                'if resuming, `run_id` should be specified.'
        run = wandb.init(
            project=args.wandb.project,
            config=to_dict(args),
            name=args.wandb.name,
            resume=args.wandb.resume,
            id=args.wandb.run_id
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
            args.dataset.num_keypoints,
            args.dataset.input_shape,
            args.model.backbone,
            is_training=True
        )
        model.build([None, *args.dataset.input_shape])
        if weight_path is not None:
            model.load_weights(weight_path)
        # train
        trainer = Trainer(args, model, logger, strategy)
        trainer.custom_loop(
            train_dist_ds,
            val_dist_ds,
            run
        )
    # evaluation
    if args.eval.use_eval:
        model = RLEModel(
            args.dataset.num_keypoints,
            args.dataset.input_shape,
            args.model.backbone,
            is_training=False
        )  # model will return heatmaps
        model.load_weights(trainer.checkpoint_prefix)
        stats = evaluate_coco(
            model,
            str(cwd.parent / args.eval.eval_pattern),
            args.dataset.num_keypoints,
            args.dataset.input_shape,
            str(cwd.parent / args.eval.coco_json)
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

        if args.wandb.use:
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
