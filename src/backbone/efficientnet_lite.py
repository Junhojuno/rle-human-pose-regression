import math
from typing import Optional, List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input


BLOCK_SETTING = [
    # repeat|kernal_size|stride|expand|input|output|se_ratio
    [1, 3, 1, 1, 32,  16,  0.25],
    [2, 3, 2, 6, 16,  24,  0.25],
    [2, 5, 2, 6, 24,  40,  0.25],
    [3, 3, 2, 6, 40,  80,  0.25],
    [3, 5, 1, 6, 80,  112, 0.25],
    [4, 5, 2, 6, 112, 192, 0.25],
    [1, 3, 1, 6, 192, 320, 0.25]
]


def conv_kernel_initializer():
    return tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_out', distribution='untruncated_normal'
    )


def dense_kernel_initializer():
    return tf.keras.initializers.VarianceScaling(
        scale=1.0 / 3.0, mode='fan_out', distribution='uniform'
    )


def round_filters(
    filters,
    multiplier: float,
    divisor: int = 8,
    min_depth: Optional[int] = None,
    skip: bool = False
):
    """Round number of filters based on depth multiplier."""
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(
        filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(
    repeats: int,
    multiplier: float,
    skip=False
):
    """Round number of filters based on depth multiplier."""
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def inverted_bottleneck(
    inputs,
    filters: int,
    expand_ratio: float,
    kernel_size: int,
    strides: int,
    survivial_prob: float = 0.8,
    name: str = 'block'
):
    """Mobile Inverted Residual Bottleneck"""
    x = inputs
    in_channels = x.shape[-1]
    prefix = name

    # expansion
    if expand_ratio != 1:
        x = layers.Conv2D(
            in_channels * expand_ratio,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name=f'{prefix}/expand/conv'
        )(x)
        x = layers.BatchNormalization(name=f'{prefix}/expand/bn')(x)
        x = layers.ReLU(max_value=6, name=f'{prefix}/expand/relu6')(x)

    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size, strides,
        depthwise_initializer=conv_kernel_initializer(),
        padding='same',
        use_bias=False,
        name=f'{prefix}/dconv'
    )(x)
    x = layers.BatchNormalization(name=f'{prefix}/bn')(x)
    x = layers.ReLU(max_value=6, name=f'{prefix}/relu6')(x)

    # output conv(projection)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer(),
        padding='same',
        use_bias=False,
        name=f'{prefix}/project/conv'
    )(x)
    x = layers.BatchNormalization(name=f'{prefix}/project/bn')(x)
    if strides == 1 and (in_channels == filters):
        if survivial_prob:
            x = layers.SpatialDropout2D(
                1 - survivial_prob,
                name=f'{prefix}/project/spatial_drop'
            )(x)
        x = layers.Add(name=f'{prefix}/project/add')([inputs, x])

    return x


def EfficientNetLite(
    input_shape: List[int],
    w_multiplier: float,
    d_multiplier: float,
    dropout_rate: float,
    num_classes: int = 1000,
    include_top: bool = False,
    name: str = 'efficientnet-lite'
):
    inputs = Input(input_shape)

    # stem part
    x = layers.Conv2D(
        filters=round_filters(32, w_multiplier, skip=True),
        kernel_size=3,
        strides=2,
        kernel_initializer=conv_kernel_initializer(),
        padding='same',
        use_bias=False,
        name='stem/conv'
    )(inputs)
    x = layers.BatchNormalization(name='stem/bn')(x)
    x = layers.ReLU(max_value=6, name='stem/relu6')(x)

    # blocks
    for i, stage_setting in enumerate(BLOCK_SETTING):
        n_reats, k, s, e, _, f, _ = stage_setting
        filters = round_filters(f, w_multiplier)
        if i != 0 and i != len(BLOCK_SETTING) - 1:
            n_reats = round_repeats(n_reats, d_multiplier)

        x = inverted_bottleneck(x, filters, e, k, s, name=f'stage{i}/block0')
        for j in range(n_reats - 1):
            x = inverted_bottleneck(
                x, filters, e, k, 1,
                name=f'stage{i}/block{j + 1}'
            )

    # head
    x = layers.Conv2D(
        filters=round_filters(1280, w_multiplier, skip=True),
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer(),
        padding='same',
        use_bias=False,
        name='head/conv'
    )(x)
    x = layers.BatchNormalization(name='head/bn')(x)
    x = layers.ReLU(max_value=6, name='head/relu6')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='head/gap')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='head/drop')(x)
        x = layers.Dense(
            num_classes,
            kernel_constraint=dense_kernel_initializer(),
            name='head/fc'
        )(x)
    return Model(inputs, x, name=name)


def EfficientNetLite0(
    input_shape: List[int] = [224, 224, 3],
    include_top=False,
    name: str = 'efficientnet-lite0'
):
    return EfficientNetLite(
        input_shape,
        w_multiplier=1.0,
        d_multiplier=1.0,
        dropout_rate=0.2,
        include_top=include_top,
        name=name
    )


def EfficientNetLite1(
    input_shape: List[int] = [240, 240, 3],
    include_top=False,
    name: str = 'efficientnet-lite1'
):
    return EfficientNetLite(
        input_shape,
        w_multiplier=1.0,
        d_multiplier=1.1,
        dropout_rate=0.2,
        include_top=include_top,
        name=name
    )


def EfficientNetLite2(
    input_shape: List[int] = [260, 260, 3],
    include_top=False,
    name: str = 'efficientnet-lite2'
):
    return EfficientNetLite(
        input_shape,
        w_multiplier=1.1,
        d_multiplier=1.2,
        dropout_rate=0.3,
        include_top=include_top,
        name=name
    )


def EfficientNetLite3(
    input_shape: List[int] = [280, 280, 3],
    include_top=False,
    name: str = 'efficientnet-lite3'
):
    return EfficientNetLite(
        input_shape,
        w_multiplier=1.2,
        d_multiplier=1.4,
        dropout_rate=0.3,
        include_top=include_top,
        name=name
    )


def EfficientNetLite4(
    input_shape: List[int] = [300, 300, 3],
    include_top=False,
    name: str = 'efficientnet-lite4'
):
    return EfficientNetLite(
        input_shape,
        w_multiplier=1.4,
        d_multiplier=1.8,
        dropout_rate=0.3,
        include_top=include_top,
        name=name
    )
