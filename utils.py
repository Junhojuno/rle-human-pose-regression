from typing import List
import tensorflow as tf
from tensorflow.python.client import device_lib
import yaml
from collections import namedtuple


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
    devices = [device.name for device in device_lib.list_local_devices() 
               if (device.name in gpu_names) and (device.memory_limit * 1e-9 > 1)]
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
