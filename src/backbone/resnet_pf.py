"""
Learning Features with Parameter-Free Layers (ICLR 2022)
https://github.com/naver-ai/PfLayer/blob/main/resnet_pf.py
"""
from functools import partial
from typing import Optional, List, Callable, Any
from tensorflow.keras import Model, Input
from tensorflow.keras import layers


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    input_size = inputs.shape[1:3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def conv_bn(
    x,
    filters,
    kernel_size=3,
    stride=1,
    groups=1,
    dilation=1,
    padding=None,
    name='conv_bn'
):
    prefix = name
    if padding is None:
        padding = (kernel_size - 1) // 2
        x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        'valid',
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
    act_layer=layers.ReLU(),
    name='conv_bn_relu'
):
    prefix = name
    if padding is None:
        padding = (kernel_size - 1) // 2
        x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        stride,
        'valid',
        groups=groups,
        dilation_rate=dilation,
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    if act_layer is not None:
        x = act_layer(x)
    return x


class ResNetPF(Model):
    expansion = 4

    def __init__(
        self,
        input_shape: list,
        block_type: str,
        n_blocks: List[int],
        n_classes: Optional[int] = 1000,
        groups: int = 1,
        base_width: Optional[int] = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        width_mult: float = 1.0,
        dilation: int = 1,
        include_top: bool = False,
        name: str = 'resnet_pf'
    ) -> None:
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation

        assert block_type in ['efficient', 'max', 'bottleneck'],\
            'block_type should be in [[efficient, max, bottleneck]'\
            'but received {}'.format(block_type)

        block = self.__define_block(block_type)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        inputs = Input(input_shape)
        x = conv_bn_relu(inputs, 64, 7, stride=2)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

        x = self.__stage(
            x,
            block,
            int(64 * width_mult),
            n_blocks[0],
            stride=1,
            dilate=False,
            name='stage1'
        )
        x = self.__stage(
            x,
            block,
            int(128 * width_mult),
            n_blocks[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            name='stage2'
        )
        x = self.__stage(
            x,
            block,
            int(256 * width_mult),
            n_blocks[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            name='stage3'
        )
        x = self.__stage(
            x,
            block,
            int(512 * width_mult),
            n_blocks[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            name='stage4'
        )
        if include_top:
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(n_classes)(x)
        super().__init__(inputs=inputs, outputs=x, name=name)

    def __stage(
        self,
        x,
        block,
        filters,
        n_blocks,
        stride: int = 1,
        dilate: bool = False,
        name: str = 'stage'
    ):
        prefix = name
        if dilate:
            self.dilation *= stride
            stride = 1

        downsample = None
        if stride != 1 \
                or (x.shape[-1] != filters * self.expansion):
            kwargs = {
                'filters': filters * self.expansion,
                'kernel_size': 1,
                'stride': stride,
            }
            downsample = partial(conv_bn, **kwargs)
        for block_i in range(n_blocks):
            downsample = downsample if block_i == 0 else None
            stride = stride if block_i == 0 else 1
            x = block(
                x,
                filters,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation
            )
        return x

    def __define_block(self, block_type):
        if block_type == 'efficient':
            block = self.__effi_block
        elif block_type == 'max':
            block = self.__max_block
        else:  # bottleneck
            block = self.__bott_block
        return block

    def __basic_block(
        self,
        x,
        filters: int,
        stride: int = 1,
        downsample: Callable = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        name: str = 'basic_block'
    ):
        x_residual = x
        out = layers.Conv2D(filters, 3, stride, use_bias=False)(x)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU()(out)
        out = layers.Conv2D(filters, 3, 1, use_bias=False)(out)
        out = layers.BatchNormalization()(out)
        if self.downsample is not None:
            x_residual = self.downsample(x)
        out = layers.Add()([out, x_residual])
        out = layers.ReLU()(out)
        return out

    def __bott_block(
        self,
        x,
        filters: int,
        stride: int = 1,
        downsample: Callable = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        name: str = 'bott_block'
    ):
        pass

    def __max_block(
        self,
        x,
        filters: int,
        stride: int = 1,
        downsample: Callable = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        name: str = 'max_block'
    ):
        pass

    def __effi_block(
        self,
        x,
        filters: int,
        stride: int = 1,
        downsample: Callable = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        name: str = 'eff_block'
    ):
        width = int(filters * (base_width / 64.)) * groups

        x_residual = x
        # reduction
        if stride == 2:
            out = conv_bn_relu(x, width, kernel_size=1)
            out = layers.AveragePooling2D(2, 2)(out)
            out = conv_bn_relu(out, width, 3, 1, groups, dilation)
        else:
            out = conv_bn_relu(x, width, kernel_size=1)
            out = layers.MaxPooling2D(3, stride, 'same')(out)

        # expansion
        out = conv_bn(out, filters * self.expansion, kernel_size=1)

        if downsample is not None:
            x_residual = downsample(x)

        out = layers.Add()([out, x_residual])
        out = layers.ReLU()(out)
        return out


def ResNet50(
    input_shape: List = [224, 224, 3],
    n_blocks: List = [3, 4, 6, 3],
    width_mult: float = 1.0,
    **kwargs: Any
) -> ResNetPF:
    return ResNetPF(
        input_shape,
        'bottleneck',
        n_blocks,
        width_mult=width_mult,
        **kwargs
    )


def ResNet50Max(
    input_shape: List = [224, 224, 3],
    n_blocks: List = [3, 4, 6, 3],
    width_mult: float = 1.0,
    **kwargs: Any
) -> ResNetPF:
    return ResNetPF(
        input_shape,
        'max',
        n_blocks,
        width_mult=width_mult,
        **kwargs
    )


def ResNet50Hybrid(
    input_shape: List = [224, 224, 3],
    n_blocks: List = [3, 4, 6, 3],
    width_mult: float = 1.0,
    **kwargs: Any
) -> ResNetPF:
    return ResNetPF(
        input_shape,
        'efficient',
        n_blocks,
        width_mult=width_mult,
        **kwargs
    )
