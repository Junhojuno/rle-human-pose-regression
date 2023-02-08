"""
학습이 완료된 모델 evaluation
"""
import tensorflow as tf
from typing import List
import numpy as np
import cv2
import json
import os
import argparse
from collections import OrderedDict
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.model import RLEModel
from src.evaluate import (
    suppress_stdout,
    load_eval_dataset,
    flip,
    flip_outputs,
    print_coco_eval
)


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        dest='backbone',
                        required=False,
                        default='resnet50')
    parser.add_argument('--weights',
                        dest='weights',
                        required=True)
    parser.add_argument('--single_coco',
                        action='store_true',
                        help='single person eval or not')
    parser.add_argument('--flip_test',
                        action='store_true',
                        help='if using flip test or not')
    return parser.parse_args()


def evaluate_coco(
    model: tf.keras.Model,
    file_pattern: str,
    num_keypoints: int = 17,
    input_shape: List = [256, 192, 3],
    coco_path: str = '',
    flip_test: bool = False
):
    with suppress_stdout():
        coco = COCO(coco_path)

    batch_size = 64
    ds = load_eval_dataset(
        file_pattern, batch_size, num_keypoints, input_shape
    )

    results = []
    for img_ids, imgs, Ms in tqdm(ds, '[coco evaluation]'):
        img_ids = img_ids.numpy()
        Ms = Ms.numpy()

        pred = model(imgs, training=False)

        if flip_test:
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

        rescored_score = np.zeros((batch_size,))
        for i in range(batch_size):
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
    return cocoEval.stats


if __name__ == '__main__':
    from pathlib import Path
    args = define_argparser()
    cwd = Path('.').resolve()
    file_pattern = str(
        cwd.parent / 'datasets/mscoco/tfrecords/val/*.tfrecord'
    )

    coco_path = str(
        cwd.parent
        / 'datasets/mscoco/annotations/person_keypoints_val2017.json'
    )

    model = RLEModel(
        17,
        [256, 192, 3],
        args.backbone,
        is_training=False
    )
    model.load_weights(args.weights)

    stats = evaluate_coco(
        model,
        file_pattern,
        17,
        [256, 192, 3],
        coco_path,
        flip_test=args.flip_test
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

    info_str = []
    for i, name in enumerate(stats_names):
        info_str.append((name, stats[i]))

    results = OrderedDict(info_str)
    print_coco_eval(results)
