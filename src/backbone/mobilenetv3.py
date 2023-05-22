from typing import List

from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.utils import get_file

from src.backbone.utils import _make_divisible, correct_pad


def MobileNetv3Large(
    input_shape: List = [256, 192, 3],
    alpha: float = 1.0,
    pretrained: bool = True,
    se_ratio: float = 0.25,
    name='mobilenet_v3_large'
):
    """backbone: MobilenetV3-Large

    Args:
        input_shape (list, optional): shape of input [H, W, C].
            Defaults to [192, 192, 3].
        alpha (float, optional): width multiplier.
            Defaults to 1.0.
        pretrained (bool, optional): whether using imagenet pretrained weights or not.
            Defaults to True.
        se_ratio (float, optional): SE block ratio. Defaults to 0.25.
        name (str, optional): name of backbone network.
            Defaults to 'mobilenet_v3_large_multi_outputs'.
    """
    def depth(d):
        return _make_divisible(d * alpha, 8)

    img_input = Input(shape=input_shape)

    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(img_input)
    x = layers.BatchNormalization(
        axis=-1, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm"
    )(x)
    x = hard_swish(x)

    # x, expansion, filters, kernel_size, stride, se_ratio, activation, id
    x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
    x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 3, depth(40), 5, 2, se_ratio, relu, 3)
    x = _inverted_res_block(x, 3, depth(40), 5, 1, se_ratio, relu, 4)
    x = _inverted_res_block(x, 3, depth(40), 5, 1, se_ratio, relu, 5)
    x = _inverted_res_block(x, 6, depth(80), 3, 2, None, hard_swish, 6)
    x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, hard_swish, 7)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, hard_swish, 8)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, hard_swish, 9)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, hard_swish, 10)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, hard_swish, 11)
    x = _inverted_res_block(x, 6, depth(160), 5, 2, se_ratio, hard_swish, 12)
    x = _inverted_res_block(x, 6, depth(160), 5, 1, se_ratio, hard_swish, 13)
    x = _inverted_res_block(x, 6, depth(160), 5, 1, se_ratio, hard_swish, 14)

    last_filter = _make_divisible(x.shape[-1] * 6, 8)
    x = layers.Conv2D(
        last_filter,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="Conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=-1, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
    )(x)
    x = hard_swish(x)
    model = Model(img_input, x, name=name)
    if pretrained and (alpha in [0.75, 1.0]):
        model_name = "{}_224_{}_float".format(
            'large', str(alpha)
        )
        file_name = "weights_mobilenet_v3_" + model_name + "_no_top_v2.h5"
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(
            file_name,
            BASE_WEIGHT_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    return model


def MobileNetv3Small(
    input_shape: List = [256, 192, 3],
    alpha: float = 1.0,
    pretrained: bool = True,
    se_ratio: float = 0.25,
    name='mobilenet_v3_small'
):
    """backbone: MobilenetV3-Small

    Args:
        input_shape (list, optional): shape of input [H, W, C].
            Defaults to [192, 192, 3].
        alpha (float, optional): width multiplier.
            Defaults to 1.0.
        pretrained (bool, optional): whether using imagenet pretrained weights or not.
            Defaults to True.
        se_ratio (float, optional): SE block ratio. Defaults to 0.25.
        name (str, optional): name of backbone network.
            Defaults to 'mobilenet_v3_small_multi_outputs'.
    """
    def depth(d):
        return _make_divisible(d * alpha, 8)

    img_input = Input(shape=input_shape)

    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(img_input)
    x = layers.BatchNormalization(
        axis=-1, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm"
    )(x)
    x = hard_swish(x)

    # x, expansion, filters, kernel_size, stride, se_ratio, activation, id
    x = _inverted_res_block(
        x, 1, depth(16), 3, 2, se_ratio, relu, 0
    )
    x = _inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 4, depth(40), 5, 2, se_ratio, hard_swish, 3)
    x = _inverted_res_block(x, 6, depth(40), 5, 1, se_ratio, hard_swish, 4)
    x = _inverted_res_block(x, 6, depth(40), 5, 1, se_ratio, hard_swish, 5)
    x = _inverted_res_block(x, 3, depth(48), 5, 1, se_ratio, hard_swish, 6)
    x = _inverted_res_block(x, 3, depth(48), 5, 1, se_ratio, hard_swish, 7)
    x = _inverted_res_block(x, 6, depth(96), 5, 2, se_ratio, hard_swish, 8)
    x = _inverted_res_block(x, 6, depth(96), 5, 1, se_ratio, hard_swish, 9)
    x = _inverted_res_block(x, 6, depth(96), 5, 1, se_ratio, hard_swish, 10)

    last_filter = _make_divisible(x.shape[-1] * 6, 8)
    x = layers.Conv2D(
        last_filter,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="Conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=-1, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
    )(x)
    x = hard_swish(x)
    model = Model(img_input, x, name=name)
    if pretrained and (alpha in [0.75, 1.0]):
        model_name = "{}_224_{}_float".format(
            'small', str(alpha)
        )
        file_name = "weights_mobilenet_v3_" + model_name + "_no_top_v2.h5"
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(
            file_name,
            BASE_WEIGHT_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    return model


BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/mobilenet_v3/"
)
WEIGHTS_HASHES = {
    "large_224_0.75_float": (
        "765b44a33ad4005b3ac83185abf1d0eb",
        "40af19a13ebea4e2ee0c676887f69a2e",
    ),
    "large_224_1.0_float": (
        "59e551e166be033d707958cf9e29a6a7",
        "07fb09a5933dd0c8eaafa16978110389",
    ),
    "large_minimalistic_224_1.0_float": (
        "675e7b876c45c57e9e63e6d90a36599c",
        "ec5221f64a2f6d1ef965a614bdae7973",
    ),
    "small_224_0.75_float": (
        "cb65d4e5be93758266aa0a7f2c6708b7",
        "ebdb5cc8e0b497cd13a7c275d475c819",
    ),
    "small_224_1.0_float": (
        "8768d4c2e7dee89b9d02b2d03d65d862",
        "d3e8ec802a04aa4fc771ee12a9a9b836",
    ),
    "small_minimalistic_224_1.0_float": (
        "99cd97fb2fcdad2bf028eb838de69e37",
        "cde8136e733e811080d9fcd8a252f7e4",
    ),
}


def _inverted_res_block(
    x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id
):
    shortcut = x
    prefix = "expanded_conv/"
    infilters = x.shape[-1]
    if block_id:
        # Expand
        prefix = "expanded_conv_{}/".format(block_id)
        x = layers.Conv2D(
            _make_divisible(infilters * expansion, 8),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand/BatchNorm",
        )(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(x, kernel_size),
            name=prefix + "depthwise/pad",
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise/BatchNorm",
    )(x)
    x = activation(x)

    if se_ratio:
        x = _se_block(
            x,
            _make_divisible(infilters * expansion, 8),
            se_ratio,
            prefix
        )

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project/BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "Add")([shortcut, x])
    return x


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + "squeeze_excite/AvgPool"
    )(inputs)
    x = layers.Conv2D(
        _make_divisible(filters * se_ratio, 8),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv",
    )(x)
    x = layers.ReLU(name=prefix + "squeeze_excite/Relu")(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv_1",
    )(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + "squeeze_excite/Mul")([inputs, x])
    return x


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])
