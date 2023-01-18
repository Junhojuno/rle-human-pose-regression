"""coco evaluation"""
import tensorflow as tf
from typing import Union, Tuple, List
import numpy as np
import cv2
import json
import os
import sys
from contextlib import contextmanager
import argparse
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.model import RLEModel
from src.transforms import transform, normalize_image


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        dest='backbone',
                        required=False,
                        default='mobilenetv2')
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


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def print_name_value(name_value, full_arch_name='ResNet50_rle', print_fn=print):
    """print out markdown format performance table
    Args:
    name_value (dict): dictionary of metric name and value
    full_arch_name (str): name of the architecture
    print_fn (print, optional): function to print results
    """
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)

    print_fn("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    print_fn("|---" * (num_values + 1) + "|")

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + "..."
    print_fn(
        "| "
        + full_arch_name
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )


def load_dataset(file_pattern, batch_size, num_keypoints=17, input_shape=[192, 192, 3]):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)

    ds = ds.map(
        lambda record: parse_example_for_cocoeval(record, num_keypoints),
        num_parallel_calls=AUTOTUNE
    )
    
    ds = ds.map(
        lambda img_id, image, bbox, keypoints: preprocess_for_cocoeval(img_id, image, bbox, keypoints,
                                                                       input_shape=input_shape),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True, 
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds


def parse_example_for_cocoeval(record, num_keypoints):
    feature_description = {
        'img_id': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([4, ], tf.float32),
        'keypoints': tf.io.FixedLenFeature([num_keypoints * 3, ], tf.float32),
    }
    example = tf.io.parse_example(record, features=feature_description)
    image = tf.io.decode_jpeg(example['image_raw'], channels=3)
    bbox = example['bbox']
    keypoints = tf.reshape(example['keypoints'], (-1, 3))
    img_id = example['img_id']
    return img_id, image, bbox, keypoints


def preprocess_for_cocoeval(img_id, img, bbox, kp, input_shape):
    kp = tf.cast(kp, tf.float32)
    
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_center = tf.cast([x + w / 2., y + h / 2.], tf.float32)

    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0] # scale with bbox
    angle = 0.
    
    # transform to the object's center
    img, M = transform(img, scale, angle, bbox_center, input_shape[:2])
    img = tf.cast(img, tf.float32)
    
    use_image_norm = True
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img = tf.cond(
        tf.math.equal(use_image_norm, True),
        lambda: normalize_image(img, means, stds),
        lambda: img
    )
    return img_id, img, M


def flip(image):
    return tf.image.flip_left_right(image)


def flip_outputs(
    outputs: tf.Tensor,
    input_width: int,
    joint_pairs: List = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
):
    pred_jts, pred_scores = outputs.mu.numpy(), outputs.maxvals.numpy()
    pred_jts[:, :, 0] = - pred_jts[:, :, 0] - 1 / input_width
    
    for pair in joint_pairs:
        dim0, dim1 = pair
        inv_pair = [dim1, dim0]
        pred_jts[:, pair, :] = pred_jts[:, inv_pair, :]
        pred_scores[:, pair, :] = pred_scores[:, inv_pair, :]
    
    outputs.mu = pred_jts
    outputs.maxvals = pred_scores
    
    return outputs


def evaluate_coco(
    model: tf.keras.Model,
    file_pattern: str,
    num_keypoints: int = 17,
    input_shape: Union[List, Tuple] = [192, 192, 3],
    coco_path: str = '',
    flip_test: bool = False
):
    with suppress_stdout():
        coco = COCO(coco_path)
    
    batch_size = 16
    ds = load_dataset(file_pattern, batch_size, num_keypoints, input_shape)

    results = []
    for img_ids, imgs, Ms in ds:
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
        
        pred_kpts = np.concatenate([pred.mu.numpy(), pred.maxvals.numpy()], axis=-1)
        kp_scores = pred.maxvals.numpy()[..., 0].copy()
        
        pred_kpts[:, :, 0] = (pred_kpts[:, :, 0] + 0.5) * input_shape[1]
        pred_kpts[:, :, 1] = (pred_kpts[:, :, 1] + 0.5) * input_shape[0]
        
        rescored_score = np.zeros((batch_size,))
        for i in range(batch_size):
            M_inv = cv2.invertAffineTransform(Ms[i])
            pred_kpts[i, :, :2] = np.matmul(M_inv[:, :2], pred_kpts[i, :, :2].T).T + M_inv[:, 2].T

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
    if args.single_coco:
        file_pattern = str(cwd.parent / 'datasets/only_coco_single_pose/val/tfrecords/*.tfrecords')
    else:
        file_pattern =  str(cwd.parent / 'datasets/only_coco/val/tfrecords/*.tfrecords')
    
    coco_path = str(cwd.parent / 'datasets/coco_dataset/annotations/person_keypoints_val2017.json')
    
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
    print_name_value(results)
