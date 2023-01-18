from typing import Optional
from tensorflow.keras import Model, Input
from tensorflow.keras import layers


def conv_bn(
    x,
    filters,
    kernel_size=3,
    stride=1,
    groups=1,
    dilation=1,
    padding=None
):
    if padding is None:
        # padding = (kernel_size - 1) // 2
        padding = 'same' if kernel_size == 3 else 'valid'
    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        padding,
        groups=groups,
        dilation_rate=dilation,
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    return x


def conv_bn_relu(
    x,
    filters,
    kernel_size=3,
    stride=1,
    groups=1,
    dilation=1,
    padding=None,
    act_layer=layers.ReLU()
):
    if padding is None:
        # padding = (kernel_size - 1) // 2
        padding = 'same' if kernel_size == 3 else 'valid'
    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        padding,
        groups=groups,
        dilation_rate=dilation,
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    if act_layer is not None:
        x = act_layer(x)
    return x


def efficient_bottleneck(
    inputs,
    filters: int ,
    stride: int = 1,
    downsample: Optional[layers.Layer] = None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
    expansion: int = 4
):
    width = int(filters * (base_width / 64.)) * groups
    
    x = inputs
    # reduction
    if stride == 2:
        x = conv_bn_relu(x, width, kernel_size=1)
        x = layers.AveragePooling2D(2, 2)(x)
        x = conv_bn_relu(x, width, 3, 1, groups, dilation)
    else:
        x = conv_bn_relu(x, width, kernel_size=1)
        x = layers.MaxPooling2D(3, stride, 'same')(x)
    
    # expansion
    x = conv_bn(x, filters * expansion, 1)
    if downsample is not None:
        x = downsample(x)
    
    x = layers.Add()([x, inputs])
    x = layers.ReLU()(x)
    return x
    
