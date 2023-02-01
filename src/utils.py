import cv2
import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.python.client import device_lib
import yaml
from collections import namedtuple
from easydict import EasyDict


def to_namedtuple(obj: dict):
    """convert dict-like object to efficient namedtuple structure
    Args:
        obj (dict): dict-like object
    Returns:
        [namedtuple]: converted namedtuple
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = to_namedtuple(value)
        return namedtuple("config", obj.keys())(**obj)

    if isinstance(obj, list):
        return [to_namedtuple(item) for item in obj]

    return obj


def load_config(config_file):
    config = yaml.safe_load(open(config_file))
    return to_namedtuple(config)


def get_available_gpu() -> List:
    """get gpus that have memory > 1GB """
    gpu_names = [gpu.name for gpu in tf.config.list_logical_devices('GPU')]
    devices = [
        device.name for device in device_lib.list_local_devices()
        if (device.name in gpu_names) and (device.memory_limit * 1e-9 > 1)
    ]
    return devices


def to_dict(args):
    """namedtuple to dict

    Args:
        args (namedtuple): configuration by namedtuple

    Returns:
        dict: configuration by dict
    """
    new_config = {}
    config = args._asdict()
    for k, v in config.items():
        if isinstance(v, str):
            new_config[k] = v
        else:
            new_config[k] = v._asdict()
    return new_config


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
    draw keypoint on original image with different color to each side's skeleton
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
