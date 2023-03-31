from typing import List
from easydict import EasyDict

import tensorflow as tf
import numpy as np

from src.transforms import generate_affine_matrix


def flip(image):
    return tf.image.flip_left_right(image)


def flip_outputs(
    outputs: EasyDict,
    input_width: int,
    joint_pairs: List = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]
    ]
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


def transform_back_to_original(pred_kpts, bbox, input_shape):
    rank = tf.rank(pred_kpts)
    assert rank == 2, "prediction should be 2"

    coords = tf.stack(
        [
            (pred_kpts[:, 0] + 0.5) * input_shape[1],
            (pred_kpts[:, 1] + 0.5) * input_shape[0]
        ],
        axis=-1
    )
    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_center = [
        x1 + w / 2,
        y1 + h / 2
    ]
    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]  # scale with bbox

    M = generate_affine_matrix(bbox_center, 0., scale, input_shape, inv=True)
    M = tf.reshape(M[:6], [2, 3])

    coords =\
        tf.transpose(
            tf.matmul(M[:, :2], coords, transpose_b=True)
        ) + M[:, -1]
    return coords


def print_coco_eval(
    name_value, full_arch_name='ResNet50_rle', print_fn=print
):
    """print out markdown format performance table
    Args:
    name_value (dict): dictionary of metric name and value
    full_arch_name (str): name of the architecture
    print_fn (print, optional): function to print results
    """
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)

    print_fn(
        "| Arch "
        + " ".join(["| {}".format(name) for name in names])
        + " |"
    )
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
