#!/bin/sh
python train.py \
    dataset=only_coco \
    scheduler=multisteplr \
    dataset.sigma=2.0 \
    augmentation.half_body_prob=0.0 \
    augmentation.rotation_factor=45 \
    train.n_gpus=1 \
    train.batch_size=32 \
    train.lr=0.001 \
    train.n_epochs=200 \
    eval.use_eval=false \
    wandb.use=false \
    wandb.project=pretraining \
    wandb.subtitle=basic_coco