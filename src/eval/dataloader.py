from typing import Dict, List

import tensorflow as tf

from src.transforms import (
    affine_transform,
    normalize_image,
    generate_target
)


def parse_eval_example(record, num_keypoints):
    feature_description = {
        'img_id': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([4, ], tf.float32),
        'keypoints': tf.io.FixedLenFeature([num_keypoints*3, ], tf.float32),
    }
    example = tf.io.parse_example(record, features=feature_description)
    image = tf.io.decode_jpeg(
        example['image_raw'],
        channels=3
    )
    bbox = example['bbox']
    keypoints = tf.reshape(example['keypoints'], (-1, 3))
    img_id = example['img_id']
    return image, bbox, keypoints, img_id


def preprocess_eval(
    image,
    bbox,
    keypoints,
    image_id: int,
    input_shape: List,
    means: List = [0.485, 0.456, 0.406],
    stds: List = [0.229, 0.224, 0.225],
):
    keypoints = tf.cast(keypoints, tf.float32)

    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_center = tf.cast([x1 + w / 2., y1 + h / 2.], tf.float32)

    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]  # scale with bbox

    image, M = affine_transform(image, bbox_center, 0., scale, input_shape)
    xy = keypoints[:, :2]
    xy = tf.transpose(tf.matmul(M[:, :2], xy, transpose_b=True)) + M[:, -1]

    # adjust visibility if coordinates are outside crop
    vis = tf.cast(keypoints[:, 2] > 0, tf.float32)  # vis==2인 경우 처리하기 위함
    vis *= tf.cast(
        (
            (xy[:, 0] >= 0) &
            (xy[:, 0] < input_shape[1]) &
            (xy[:, 1] >= 0) &
            (xy[:, 1] < input_shape[0])
        ),
        tf.float32
    )
    keypoints = tf.concat([xy, tf.expand_dims(vis, axis=1)], axis=1)

    image = tf.cast(image, tf.float32)
    image = normalize_image(image, means, stds)
    target = generate_target(keypoints, input_shape)
    return image, target, bbox, image_id


def load_eval_dataset(
    file_pattern: str,
    batch_size: int,
    num_keypoints: int,
    input_shape: List
) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda record: parse_eval_example(record, num_keypoints),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(
        lambda image, bbox, keypoints, image_id: preprocess_eval(
            image, bbox, keypoints, image_id, input_shape
        ),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True,
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds
