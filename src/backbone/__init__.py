from typing import List, Optional

from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from src.backbone.efficientnet import EfficientNetB0
from src.backbone.ghostnetv2 import GhostNetV2
from src.backbone.resnet_pf import ResNet50Hybrid
from src.backbone.efficientnet_lite import EfficientNetLite0
from src.backbone.mobilenetv3 import MobileNetv3Large, MobileNetv3Small
from src.backbone.shufflenetv2 import ShuffleNetV2


def build_backbone(
    backbone_type: Optional[str] = 'mobilenetv2',
    input_shape: Optional[List] = [256, 192, 3]
) -> Model:
    if backbone_type.startswith('resnet50'):
        backbone = ResNet50(include_top=False, input_shape=input_shape)
    elif backbone_type.startswith('mobilenetv2'):
        backbone = MobileNetV2(include_top=False, input_shape=input_shape)
    elif backbone_type.startswith('resnet50-pf'):
        backbone = ResNet50Hybrid(include_top=False, input_shape=input_shape)
    elif backbone_type.startswith('efficientnet'):
        backbone = EfficientNetB0(include_top=False, input_shape=input_shape)
    elif backbone_type.startswith('ghostnetv2'):
        backbone = GhostNetV2(
            include_top=False,
            input_shape=input_shape,
            width_multiplier=1.0,
        )
    elif backbone_type.startswith('efficientnet-lite'):
        backbone = EfficientNetLite0(
            include_top=False,
            input_shape=input_shape
        )
    elif backbone_type.startswith('mobilenetv3-large'):
        alpha = float(backbone_type.split('_')[1])
        backbone = MobileNetv3Large(
            input_shape=input_shape,
            alpha=alpha,
            name=f'mobilenetv3_large_{alpha}'
        )
    elif backbone_type.startswith('mobilenetv3-small'):
        backbone = MobileNetv3Small(
            input_shape=input_shape,
            alpha=1.0,
            name='mobilenetv3_small_1.0'
        )
    elif backbone_type == 'shufflenetv2':
        backbone = ShuffleNetV2(
            input_shape=input_shape,
            model_size='1.0x',
            name='shufflenetv2'
        )
    else:
        raise ValueError(f'{backbone_type} is wrong. check backbone_type')
    return backbone
