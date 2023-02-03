from typing import Tuple
import random
import tensorflow as tf
import albumentations as A


def random_src_color() -> Tuple[int, int, int]:
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


light = A.Compose(
    [
        A.HueSaturationValue(p=1),
        A.RandomContrast(p=1),
        A.RandomBrightness(limit=0.25, p=1),
    ],
    p=1
)

medium = A.Compose(
    [
        A.CLAHE(p=1),
        A.RandomBrightness(limit=0.3, p=1),
    ],
    p=1
)


strong = A.Compose(
    [
        A.ChannelShuffle(p=1),
        A.RandomBrightness(limit=0.4, p=1),
    ], p=0.5
)


def get_albumentation():
    a_transform = A.Compose([
        A.OneOf(
            [
                light,
                medium,
                strong
            ],
            p=0.5
        ),
        A.CoarseDropout(
            max_height=30,
            max_width=30,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=random_src_color(),
            p=0.2
        ),
        A.ISONoise(p=0.5, color_shift=(0.01, 0.2)),
        A.GaussNoise(p=0.25),
        A.ImageCompression(quality_lower=50, p=0.25),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1),
                A.RingingOvershoot(p=1),
            ],
            p=0.25
        ),
        # extra noise
        A.OneOf(
            [
                A.RandomGamma(p=1),
                A.FancyPCA(p=1, alpha=3.0),
                A.MultiplicativeNoise(
                    p=1, multiplier=(0.5, 1.1), per_channel=True
                ),
                A.RandomToneCurve(p=1, scale=0.5),
            ],
            p=0.1
        ),
    ])
    return a_transform


def aug_fn(img):
    """augmentation using albumentations library"""
    data = {"image": img}
    aug_data = get_albumentation()(**data)
    return aug_data["image"]


@tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8)])
def apply_albumentaion(img) -> tf.Tensor:
    """apply augmentation using albumentations library"""
    img = tf.numpy_function(func=aug_fn, inp=[img], Tout=tf.uint8)
    return img
