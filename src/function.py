import os
import json
import numpy as np
from typing import List
import cv2
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf

from src.metrics import calc_coord_accuracy
from src.evaluate import (
    flip,
    flip_outputs,
    suppress_stdout,
    print_coco_eval,
    STATS_NAMES
)


@tf.function
def train(inputs, model, criterion, optimizer, args):
    images, targets = inputs

    with tf.GradientTape() as tape:
        pred = model(images, mu_g=targets[..., :2], training=True)
        loss = criterion(targets, pred)
        loss = tf.math.reduce_mean(loss)
        loss += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )
    acc = calc_coord_accuracy(
        targets, pred, args.DATASET.COMMON.OUTPUT_SHAPE
    )
    return loss, acc


def validate(
    model,
    eval_ds,
    input_shape: List,
    coco_path: str = '',
    use_flip: bool = True
):
    with suppress_stdout():
        coco = COCO(coco_path)

    results = []
    for img_ids, imgs, Ms in eval_ds:
        img_ids = img_ids.numpy()
        Ms = Ms.numpy()

        pred = model(imgs, training=False)

        if use_flip:
            imgs_fliped = flip(imgs)
            pred_fliped = model(imgs_fliped, training=False)
            pred_fliped = flip_outputs(pred_fliped, input_shape[1])
            for k in pred.keys():
                if isinstance(pred[k], list):
                    continue
                if pred[k] is not None:
                    pred[k] = (pred[k] + pred_fliped[k]) / 2

        pred_kpts = np.concatenate(
            [pred.mu.numpy(), pred.maxvals.numpy()],
            axis=-1
        )
        kp_scores = pred.maxvals.numpy()[..., 0].copy()

        pred_kpts[:, :, 0] = (pred_kpts[:, :, 0] + 0.5) * input_shape[1]
        pred_kpts[:, :, 1] = (pred_kpts[:, :, 1] + 0.5) * input_shape[0]

        rescored_score = np.zeros((imgs.numpy().shape[0],))
        for i in range(imgs.numpy().shape[0]):
            M_inv = cv2.invertAffineTransform(Ms[i])
            pred_kpts[i, :, :2] = \
                np.matmul(
                    M_inv[:, :2], pred_kpts[i, :, :2].T
                ).T \
                + M_inv[:, 2].T

            # rescore
            score_mask = kp_scores[i] > 0.2  # score threshold in validation
            if np.sum(score_mask) > 0:
                rescored_score[i] = np.mean(kp_scores[i][score_mask])

            results.append(dict(image_id=int(img_ids[i]),
                                category_id=1,
                                keypoints=pred_kpts[i].reshape(-1).tolist(),
                                score=float(rescored_score[i])))

    result_path = '{}/{}_{}.json'.format('models', 'rle_model', 'val')
    os.makedirs('./models', exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(results, f)

    with suppress_stdout():
        result = coco.loadRes(result_path)
        cocoEval = COCOeval(coco, result, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    info_str = []
    for i, name in enumerate(STATS_NAMES):
        info_str.append((name, cocoEval.stats[i]))

    results = OrderedDict(info_str)
    print_coco_eval(results)
    return cocoEval.stats  # return AP/AR all
