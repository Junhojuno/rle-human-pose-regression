"""
Creates a ShuffleNetV2 Model as defined in:
ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    - https://arxiv.org/pdf/1807.11164v1.pdf
    - https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2

"""
import tensorflow as tf
from keras import Model, Input, Sequential
from keras import layers


def channel_shuffle(x):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H, W, 2, C // 2])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, H, W, C])
    x1, x2 = tf.split(x, 2, axis=-1)
    return x1, x2


def _block(x, middle_filters, out_filters, kernel_size, stride):
    # channel split
    if stride == 1:
        x1, x2 = channel_shuffle(x)
    else:
        x1, x2 = x, x

    # main branch
    in_filter = x2.shape[-1]
    x2 = main_branch_ops(
        x2,
        middle_filters,
        out_filters - in_filter,
        kernel_size,
        stride
    )
    # identity/projection branch
    if stride == 2:
        x1 = proj_branch_ops(x1, kernel_size, stride)

    return layers.Concatenate()([x1, x2])


def main_branch_ops(x, mid_c, out_c, kernel_size, stride):
    return Sequential(
        [
            layers.Conv2D(mid_c, 1, 1, 'valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.ZeroPadding2D(kernel_size // 2),
            layers.DepthwiseConv2D(
                kernel_size, stride, 'valid', use_bias=False
            ),
            layers.BatchNormalization(),
            layers.Conv2D(out_c, 1, 1, 'valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
    )(x)


def proj_branch_ops(x, kernel_size, stride):
    in_filter = x.shape[-1]
    return Sequential(
        [
            layers.ZeroPadding2D(kernel_size // 2),
            layers.DepthwiseConv2D(
                kernel_size, stride, 'valid', use_bias=False
            ),
            layers.BatchNormalization(),
            layers.Conv2D(in_filter, 1, 1, 'valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
    )(x)


def first_conv_bn(x):
    return Sequential(
        [
            layers.ZeroPadding2D(1),
            layers.Conv2D(24, 3, 2, 'valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
    )(x)


def last_conv_bn(x, filters):
    return Sequential(
        [
            layers.Conv2D(filters, 1, 1, 'valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
    )(x)


def stage(x, filters, n_repeats=1):
    for i in range(n_repeats):
        x = _block(x, filters // 2, filters, 3, stride=2 if i == 0 else 1)
    return x


def ShuffleNetV2(input_shape, model_size='1.0x', name='shufflenetv2'):
    inputs = Input(input_shape)
    if model_size == '0.5x':
        stage_filters = [48, 96, 192, 1024]
    elif model_size == '1.0x':
        stage_filters = [116, 232, 464, 1024]
    elif model_size == '1.5x':
        stage_filters = [176, 352, 704, 1024]
    elif model_size == '2.0x':
        stage_filters = [244, 488, 976, 2048]
    else:
        raise NotImplementedError

    x = first_conv_bn(inputs)
    x = layers.MaxPooling2D(3, 2, 'same')(x)

    x = stage(x, stage_filters[0], 4)  # stage 1
    x = stage(x, stage_filters[1], 8)  # stage 2
    x = stage(x, stage_filters[2], 4)  # stage 3

    x = last_conv_bn(x, stage_filters[-1])
    return Model(inputs, x, name=f'{name}_{model_size}')
