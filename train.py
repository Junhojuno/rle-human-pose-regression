"""Custom training pipeline"""
from pathlib import Path
import numpy as np
import tensorflow as tf
import logging

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import wandb

from model import RegressFlow
from trainer import Trainer
# from core.eval.simple_evaluator import SimpleEvaluator
from dataset import load_dataset

from utils import get_available_gpu, to_dict, to_namedtuple

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

    if args.eval.use_eval:
        # eval_ds = load_pose_dataset(
        #     str(cwd.parent / args.eval.eval_pattern),
        #     args.eval.batch_size * strategy.num_replicas_in_sync,
        #     args,
        #     mode='val',
        #     use_aug=False,
        #     use_transform=True,
        #     use_album_aug=False
        # )
        eval_ds = None
    else:
        eval_ds = None

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
        model = RegressFlow(
            args.dataset.num_keypoints,
            args.dataset.input_shape,
            is_training=True
        )
        model.build([None, *args.dataset.input_shape])
        # train
        trainer = Trainer(args, model, logger, strategy)
        trainer.custom_loop(
            train_dist_ds,
            val_dist_ds,
            run
        )
    # # evaluation
    # if eval_ds is not None:
    #     model = build_baseline(
    #         args.dataset.input_shape,
    #         args.dataset.num_keypoints,
    #         args.model.backbone,
    #         weight_path=trainer.checkpoint_prefix,  # load best model
    #         is_training=True  # this doesn't mean that weights are updated.
    #     )  # model will return heatmaps
    #     if args.eval.type == 'weelo':
    #         evaluator = SimpleEvaluator(
    #             args.dataset.output_shape,
    #             args.dataset.num_keypoints
    #         )  # initialize in every eval
    #         eval_results = evaluator.evaluate(model, eval_ds)
    #         eval_table = wandb.Table(
    #             data=[eval_results],
    #             columns=args.eval.cols
    #         )
    #     elif args.eval.type == 'coco':
    #         # TODO: evaluation with COCO
    #         pass

    #     run.log({'eval': eval_table})


if __name__ == '__main__':
    main()
