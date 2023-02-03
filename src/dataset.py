from typing import Dict
import tensorflow as tf

from src.transforms import (
    parse_example,
    preprocess
)


def load_dataset(
    file_pattern: str,
    batch_size: int,
    args: Dict,
    mode: str,
    use_aug: bool
) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)

    if mode == 'train':
        ds = ds.shuffle(buffer_size=30000, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda record: parse_example(record, args.DATASET.COMMON.K),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(
        lambda image, bbox, keypoints: preprocess(
            image, bbox, keypoints,
            use_image_norm=args.DATASET.COMMON.IMAGE_NORM,
            means=args.DATASET.COMMON.MEANS,
            stds=args.DATASET.COMMON.STDS,
            scale_factor=args.AUG.SCALE_FACTOR,
            rotation_prob=args.AUG.ROT_PROB,
            rotation_factor=args.AUG.ROT_FACTOR,
            flip_prob=args.AUG.FLIP_PROB,
            flip_kp_indices=args.AUG.KP_FLIP,
            half_body_prob=args.AUG.HALF_BODY_PROB,
            half_body_min_kp=args.AUG.HALF_BODY_MIN_KP,
            kpt_upper=args.AUG.KP_UPPER,
            input_shape=args.DATASET.COMMON.INPUT_SHAPE,
            album=args.AUG.ALBUM,
            use_aug=use_aug,
        ),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True,
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds
