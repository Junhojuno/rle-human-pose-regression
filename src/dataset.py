from typing import Dict
import tensorflow as tf

from transforms import (
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
        lambda record: parse_example(record, num_keypoints=args.dataset.num_keypoints),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(
        lambda image, bbox, keypoints: preprocess(
            image, bbox, keypoints,
            use_image_norm=args.dataset.use_norm,
            means=args.dataset.means,
            stds=args.dataset.stds,
            scale_factor=args.augmentation.scale_factor,
            rotation_prob=args.augmentation.rotation_prob,
            rotation_factor=args.augmentation.rotation_factor,
            flip_prob=args.augmentation.flip_prob,
            flip_kp_indices=args.augmentation.kp_flip,
            half_body_prob=args.augmentation.half_body_prob,
            half_body_min_kp=args.augmentation.half_body_min_kp,
            kpt_upper=args.augmentation.kp_upper,
            input_shape=args.dataset.input_shape,
            use_aug=use_aug,
        ),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True, 
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds
