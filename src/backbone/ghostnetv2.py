"""
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py"""
import math
from typing import Optional, List
from tensorflow.keras import Model, Input
from tensorflow.keras import layers

from src.backbone.utils import _make_divisible, get_weights_from_remote


def hard_sigmoid(x, name: str = 'hard_sigmoid'):
    return layers.ReLU(max_value=6., name=name)(x + 3.) / 6.


def se_block(
    inputs,
    filters: int,
    se_ratio: float = 0.25,
    name: str = 'se'
):
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=f'{name}/gap'
    )(inputs)
    x = layers.Conv2D(
        _make_divisible(filters * se_ratio, 4),
        kernel_size=1,
        padding='same',
        name=f'{name}/conv1'
    )(x)
    x = layers.ReLU(name=f'{name}/relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=f'{name}/conv2'
    )(x)
    x = hard_sigmoid(x, name=f'{name}/hard_sigmoid')
    x = layers.Multiply(name=f'{name}/mul')([inputs, x])
    return x


def conv_bn_act(
    x,
    filters: int,
    kernel_size: int,
    stride: int = 1,
    name: str = 'conv_block'
):
    x = layers.ZeroPadding2D(
        kernel_size // 2,
        name=f'{name}/pad'
    )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        'valid',
        use_bias=False,
        name=f'{name}/conv'
    )(x)
    x = layers.BatchNormalization(name=f'{name}/bn')(x)
    x = layers.ReLU(name=f'{name}/relu')(x)
    return x


def cheap_operation(
    x,
    kernel_size: int,
    stride: int,
    use_relu: bool = False,
    name: str = 'cheap_ops'
):
    x = layers.ZeroPadding2D(
        kernel_size // 2,
        name=f'{name}/pad'
    )(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        stride,
        'valid',
        use_bias=False,
        name=f'{name}/dw'
    )(x)
    x = layers.BatchNormalization(name=f'{name}/bn')(x)
    if use_relu:
        x = layers.ReLU(name=f'{name}/relu')(x)
    return x


def dfc_module(
    x,
    filters: int,
    kernel_size: int,
    stride: int,
    name: str = 'dfc'
):
    """decoupled-fully-connected

    to capture the long-range information
    """
    x = layers.ZeroPadding2D(
        kernel_size // 2,
        name=f'{name}/pad1'
    )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        'valid',
        use_bias=False,
        name=f'{name}/conv'
    )(x)
    x = layers.BatchNormalization(name=f'{name}/bn1')(x)
    x = layers.ZeroPadding2D((0, 2), name=f'{name}/pad2')(x)
    x = layers.DepthwiseConv2D(
        kernel_size=(1, 5),
        strides=1,
        padding='valid',
        use_bias=False,
        name=f'{name}/dw1'
    )(x)
    x = layers.BatchNormalization(name=f'{name}/bn2')(x)
    x = layers.ZeroPadding2D((2, 0), name=f'{name}/pad3')(x)
    x = layers.DepthwiseConv2D(
        kernel_size=(5, 1),
        strides=1,
        padding='valid',
        use_bias=False,
        name=f'{name}/dw2'
    )(x)
    x = layers.BatchNormalization(name=f'{name}/bn3')(x)
    return x


def ghost_module_v2(
    x,
    filters: int,
    kernel_size: int = 1,
    ratio: int = 2,
    dw_size: int = 3,
    stride: int = 1,
    use_relu: bool = True,
    mode: str = 'original',
    name: str = 'ghost_module'
):
    init_channels = int(math.ceil(filters / ratio))
    # primary conv
    x1 = layers.Conv2D(
        init_channels,
        kernel_size,
        stride,
        'valid',
        use_bias=False,
        name=f'{name}/conv'
    )(x)
    x1 = layers.BatchNormalization(name=f'{name}/bn')(x1)
    if use_relu:
        x1 = layers.ReLU(name=f'{name}/relu')(x1)

    x2 = cheap_operation(x1, dw_size, 1, use_relu, name=f'{name}/cheap')
    out = layers.Concatenate(name=f'{name}/concat')([x1, x2])
    # out = out[..., :filters]

    if mode == 'attn':
        res = layers.AveragePooling2D(2, 2, name=f'{name}/pool')(x)
        res = dfc_module(res, filters, kernel_size, stride, name=f'{name}/dfc')
        res = layers.Activation('sigmoid', name=f'{name}/sigmoid')(res)
        res = layers.Resizing(
            out.shape[1],
            out.shape[2],
            'nearest',
            name=f'{name}/resize'
        )(res)
        return layers.Multiply(name=f'{name}/mul')([out, res])

    return out


def ghost_bottleneck_v2(
    x,
    mid_filters: int,
    out_filters: int,
    dw_size: int = 3,
    stride: int = 1,
    se_ratio: float = 0.,
    layer_id: Optional[int] = None
):
    prefix = f'bottleneck_{layer_id:02d}'
    has_se = se_ratio is not None and se_ratio > 0.
    mode = 'original' if layer_id <= 1 else 'attn'
    use_shortcut = (x.shape[-1] != out_filters) or (stride != 1)

    x_residual = x
    if use_shortcut:
        x_residual = layers.ZeroPadding2D(
            dw_size // 2,
            name=f'{prefix}/short/pad'
        )(x_residual)
        x_residual = layers.DepthwiseConv2D(
            dw_size,
            stride,
            'valid',
            use_bias=False,
            name=f'{prefix}/short/dw'
        )(x_residual)
        x_residual = layers.BatchNormalization(
            name=f'{prefix}/short/bn1'
        )(x_residual)
        x_residual = layers.Conv2D(
            out_filters,
            1,
            1,
            'valid',
            use_bias=False,
            name=f'{prefix}/short/conv'
        )(x_residual)
        x_residual = layers.BatchNormalization(
            name=f'{prefix}/short/bn2'
        )(x_residual)

    x = ghost_module_v2(
        x,
        mid_filters,
        use_relu=True,
        mode=mode,
        name=f'{prefix}/ghost1'
    )
    if stride > 1:
        x = layers.ZeroPadding2D(dw_size // 2, name=f'{prefix}/down/pad')(x)
        x = layers.DepthwiseConv2D(
            dw_size,
            stride,
            'valid',
            use_bias=False,
            name=f'{prefix}/down/dw'
        )(x)
        x = layers.BatchNormalization(name=f'{prefix}/down/bn')(x)

    if has_se:
        x = se_block(x, mid_filters, se_ratio, name=f'{prefix}/se')

    x = ghost_module_v2(
        x,
        out_filters,
        use_relu=False,
        mode='original',
        name=f'{prefix}/ghost2'
    )
    return layers.Add(name=f'{prefix}/short/add')([x, x_residual])


def GhostNetV2(
    input_shape: List,
    width_multiplier: float = 1.0,
    dropout: float = 0.,
    num_classes: int = 1000,
    include_top: bool = False,
    name='ghostnetv2'
) -> Model:
    def depth(d):
        return _make_divisible(d * width_multiplier, 4)

    out_filters = _make_divisible(16 * width_multiplier, 4)
    inputs = Input(input_shape)

    # stem layer
    x = layers.Conv2D(
        out_filters,
        3,
        2,
        'same',
        use_bias=False,
        name='stem/conv'
    )(inputs)
    x = layers.BatchNormalization(name='stem/bn')(x)
    x = layers.ReLU(name='stem/relu')(x)

    # blocks
    x = ghost_bottleneck_v2(x, depth(16), depth(16), 3, 1, 0.0, layer_id=0)
    x = ghost_bottleneck_v2(x, depth(48), depth(24), 3, 2, 0.0, layer_id=1)
    x = ghost_bottleneck_v2(x, depth(72), depth(24), 3, 1, 0.0, layer_id=2)
    x = ghost_bottleneck_v2(x, depth(72), depth(40), 5, 2, 0.25, layer_id=3)
    x = ghost_bottleneck_v2(x, depth(120), depth(40), 5, 1, 0.25, layer_id=4)
    x = ghost_bottleneck_v2(x, depth(240), depth(80), 3, 2, 0.0, layer_id=5)
    x = ghost_bottleneck_v2(x, depth(200), depth(80), 3, 1, 0.0, layer_id=6)
    x = ghost_bottleneck_v2(x, depth(184), depth(80), 3, 1, 0.0, layer_id=7)
    x = ghost_bottleneck_v2(x, depth(184), depth(80), 3, 1, 0.0, layer_id=8)
    x = ghost_bottleneck_v2(x, depth(480), depth(112), 3, 1, 0.25, layer_id=9)
    x = ghost_bottleneck_v2(x, depth(672), depth(112), 3, 1, 0.25, layer_id=10)
    x = ghost_bottleneck_v2(x, depth(672), depth(160), 5, 2, 0.25, layer_id=11)
    x = ghost_bottleneck_v2(x, depth(960), depth(160), 5, 1, 0.0, layer_id=12)
    x = ghost_bottleneck_v2(x, depth(960), depth(160), 5, 1, 0.25, layer_id=13)
    x = ghost_bottleneck_v2(x, depth(960), depth(160), 5, 1, 0.0, layer_id=14)
    x = ghost_bottleneck_v2(x, depth(960), depth(160), 5, 1, 0.25, layer_id=15)

    x = conv_bn_act(x, depth(960), 1, 1, name='final/conv_block')

    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True, name='final/gap')(x)
        x = layers.Conv2D(
            1280,
            1,
            1,
            'valid',
            use_bias=True,
            name='final/conv'
        )(x)
        x = layers.ReLU(name='final/relu')(x)
        x = layers.Flatten(name='final/flatten')(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name='final/dropout')(x)
        x = layers.Dense(num_classes, name='final/classifier')(x)
        return Model(inputs, x, name=name)

    model = Model(inputs, x, name=name)
    if width_multiplier == 1.:
        filename = 'ghostnetv2_1.0_imagenet.h5'
        pretrained_weights = get_weights_from_remote('ghostnetv2', filename)
        model.load_weights(pretrained_weights, by_name=True)
        print(f"====== GhostNetV2 load weights : {pretrained_weights} ======")
    return model
