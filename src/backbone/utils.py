from pathlib import Path


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_weights_from_remote(backbone_name: str, filename: str) -> str:
    """
    backbone_names:
        GhostNetV2 -> ghostnetv2_1.0_imagenet.h5
        TinyNetA -> tinynet_a_imagenet.h5
    """
    # cwd has been changed because of hydra.
    # so, more reliable cwd is needed.
    cwd = Path(__file__)  # alycepose-lite/src/model/backbone/ghostnetv2.py
    weights_path =\
        cwd.parents[2] / \
        f'results/pretrained_backbone/{backbone_name}/{filename}'
    return str(weights_path)


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
