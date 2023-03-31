import sys
import json
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, List
import numpy as np
from collections import OrderedDict

import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.utils import (
    flip,
    flip_outputs,
    transform_back_to_original,
    print_coco_eval
)

# TODO: Ground-Truth로 transform_back이 되는지 시각화해서 확인해보기!


def eval_coco(
    model,
    eval_ds,
    batch_size: int = 64,
    input_shape: List = [256, 192, 3],
    use_flip: bool = True
) -> Tuple[float, Dict]:
    """evaluate model with GT"""
    cwd = Path(__file__).resolve().parent.parent.parent
    eval_ds = tqdm(eval_ds, '[RLE/Evaluation]')

    kpt_json = []
    for images, _, bboxes, image_ids in eval_ds:
        pred_outputs = model(images, training=False)

        if use_flip:
            images = flip(images)
            pred_outputs_flipped = model(images, training=False)
            pred_outputs_flipped = flip_outputs(
                pred_outputs_flipped, input_shape[1]
            )
            for k in pred_outputs.keys():
                if isinstance(pred_outputs[k], list):
                    continue
                if pred_outputs[k] is not None:
                    pred_outputs[k] = (
                        pred_outputs[k] + pred_outputs_flipped[k]
                    ) / 2

        pred_kpts = pred_outputs.mu
        pred_scores = pred_outputs.maxvals
        for b in range(batch_size):
            pred_kpt = pred_kpts[b, :, :2]
            pred_score = pred_scores[b, :, :]
            # pred_kpt = true_kpts[b, :, :2]
            # pred_score = true_kpts[b, :, 2:]

            bbox = bboxes[b]
            pred_kpt = transform_back_to_original(
                pred_kpt, bbox, input_shape
            )
            keypoints = tf.concat([pred_kpt, pred_score], axis=-1)
            keypoints = keypoints.numpy().reshape(-1).tolist()

            data = dict()
            data['bbox'] = bbox.numpy().tolist()
            data['image_id'] = int(image_ids.numpy()[b])
            data['score'] = float(
                np.mean(pred_score) + np.amax(pred_score)
            )
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    result_path = str(cwd / 'eval_model_result.json')
    with open(result_path, 'w') as f:
        json.dump(kpt_json, f)

    class NullWriter(object):

        def write(self, arg):
            pass

    ann_file = str(
        cwd.parent
        / 'datasets/mscoco/annotations/person_keypoints_val2017.json'
    )

    nullwrite = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwrite  # disable output

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(result_path)

    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    sys.stdout = oldstdout  # enable output

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = OrderedDict()
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]

    print_coco_eval(info_str, 'RLE_Model', print)

    return info_str['AP'], info_str
