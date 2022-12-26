#!/bin/sh
python train.py \
    dataset=only_coco \
    scheduler=multisteplr \
    dataset.sigma=2.0 \
    augmentation.scale_factor=0.25 \
    augmentation.half_body_prob=0.3 \
    augmentation.rotation_factor=45 \
    train.n_gpus=1 \
    train.batch_size=32 \
    train.lr=0.001 \
    train.n_epochs=270 \
    eval.use_eval=true \
    wandb.use=true \
    wandb.project=rle_pretraining \
    wandb.subtitle=basic_coco