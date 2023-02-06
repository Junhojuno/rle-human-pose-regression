import cv2
import numpy as np
from typing import List
import tensorflow as tf
import yaml
from easydict import EasyDict
import logging
import tempfile
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


KEYPOINT_INDEX_TO_NAME = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


def visualize_with_RGB(
    image,
    keypoints,
    threshold: float = 0.3,
    alpha: float = 0.5
):
    """
    draw keypoint on original image
    with different color to each side's skeleton
        - right: red
        - intermediate: green
        - left: blue
    """
    kepoint_pairs = [
        [5, 6], [6, 12], [12, 11], [11, 5],
        [5, 7], [7, 9], [11, 13], [13, 15],
        [6, 8], [8, 10], [12, 14], [14, 16]
    ]

    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    blue_color = (255, 0, 0)

    # keypoint
    for i in range(len(keypoints)):
        if keypoints[i, 2] > threshold:
            if KEYPOINT_INDEX_TO_NAME[i].startswith('left_'):
                image = cv2.circle(
                    image,
                    (int(keypoints[i][0]), int(keypoints[i][1])),
                    4,
                    blue_color,
                    thickness=4
                )
            elif KEYPOINT_INDEX_TO_NAME[i].startswith('right_'):
                image = cv2.circle(
                    image,
                    (int(keypoints[i][0]), int(keypoints[i][1])),
                    4,
                    red_color,
                    thickness=1,
                    lineType=cv2.FILLED
                )
            else:  # nose
                image = cv2.circle(
                    image,
                    (int(keypoints[i][0]), int(keypoints[i][1])),
                    4,
                    green_color,
                    thickness=1,
                    lineType=cv2.FILLED
                )

    # skeleton
    for p in kepoint_pairs:
        if keypoints[p[0], 2] > threshold and keypoints[p[1], 2] > threshold:
            is_edge_1_left = KEYPOINT_INDEX_TO_NAME[p[0]].startswith('left_')
            is_edge_2_left = KEYPOINT_INDEX_TO_NAME[p[1]].startswith('left_')
            if is_edge_1_left and is_edge_2_left:
                image = cv2.line(
                    image,
                    tuple(np.int32(np.round(keypoints[p[0], :2]))),
                    tuple(np.int32(np.round(keypoints[p[1], :2]))),
                    color=blue_color,  # blue
                    thickness=2
                )
            elif is_edge_1_left or is_edge_2_left:
                image = cv2.line(
                    image,
                    tuple(np.int32(np.round(keypoints[p[0], :2]))),
                    tuple(np.int32(np.round(keypoints[p[1], :2]))),
                    color=green_color,  # green
                    thickness=2
                )
            else:  # right
                image = cv2.line(
                    image,
                    tuple(np.int32(np.round(keypoints[p[0], :2]))),
                    tuple(np.int32(np.round(keypoints[p[1], :2]))),
                    color=red_color,  # red
                    thickness=2
                )
    image = cv2.addWeighted(image, alpha, image, 1 - alpha, 0)
    return image


def parse_yaml(yaml_file: str) -> EasyDict:
    data = yaml.load(
        open(yaml_file, 'r'), Loader=yaml.Loader
    )
    args = EasyDict(data)
    return args


def get_flops(
    model: tf.keras.Model,
    input_shape: List,
    write_path: str = tempfile.NamedTemporaryFile().name
) -> float:
    """return GFLOPS"""
    forward_pass = tf.function(
        model.call,
        input_signature=[
            tf.TensorSpec(shape=(1, *input_shape))
        ]
    )
    opts = ProfileOptionBuilder.float_operation()
    if write_path:
        opts['output'] = 'file:outfile={}'.format(write_path)
    graph_info = profile(
        forward_pass.get_concrete_function().graph,
        options=opts
    )
    return graph_info.total_float_ops / 2 / 1e9


def get_log_template(
    epoch: int,
    total_time: int,
    train_time: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float
):
    log_text = 'Epoch: {epoch:03d} - {total_time}s[{train_time}s] '\
               '| Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} '\
               '| Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f} '\
               '| LR: {lr}'
    log_text.format(
        epoch=epoch,
        total_time=total_time,
        train_time=train_time,
        t_loss=train_loss,
        t_acc=train_acc,
        v_loss=val_loss,
        v_acc=val_acc,
        lr=lr
    )
    return log_text


def get_logger(log_file_path, name='rle-pose'):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s line:%(lineno)d]:%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
