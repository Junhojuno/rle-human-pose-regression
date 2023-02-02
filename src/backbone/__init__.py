from typing import List, Optional
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from src.backbone.resnet_pf import ResNet50Hybrid


def build_backbone(
    backbone_type: Optional[str] = 'mobilenetv2',
    input_shape: Optional[List] = [256, 192, 3]
) -> Model:
    if backbone_type == 'resnet50':
        backbone = ResNet50(include_top=False, input_shape=input_shape)
    elif backbone_type == 'mobilenetv2':
        backbone = MobileNetV2(include_top=False, input_shape=input_shape)
    elif backbone_type == 'resnet50-pf':
        backbone = ResNet50Hybrid(include_top=False, input_shape=input_shape)
    elif backbone == 'efficientnet':
        backbone = EfficientNetB0(include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f'{backbone_type} is wrong. check backbone_type')
    return backbone
