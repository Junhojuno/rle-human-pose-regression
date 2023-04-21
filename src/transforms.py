import math
from typing import List
import tensorflow as tf
import tensorflow_addons as tfa

from src.aug import apply_albumentaion


def parse_example(record, num_keypoints):
    feature_description = {
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
    return image, bbox, keypoints


def affine_transform(
    image,
    bbox_center,
    angle: float,
    scale: float,
    input_shape: List,
):
    """return transformed image and Matrix"""
    M = generate_affine_matrix(
        bbox_center,
        angle,
        scale,
        [input_shape[0] - 1, input_shape[1] - 1],
        inv=False
    )
    M = tf.reshape(M[:6], [2, 3])

    transforms = generate_affine_matrix(
        bbox_center,
        angle,
        scale,
        [input_shape[0] - 1, input_shape[1] - 1],
        inv=True
    )
    transformed_image = tfa.image.transform(
        tf.expand_dims(image, 0),
        tf.expand_dims(transforms, 0),
        output_shape=input_shape[:2]
    )
    transformed_image = tf.squeeze(transformed_image, 0)
    return transformed_image, M


def generate_affine_matrix(
    bbox_center,
    angle: float,
    scale: float,
    input_shape: List,
    inv: bool = False
):
    crop_mat = generate_crop_matrix(bbox_center, scale, input_shape, inv=inv)
    resize_mat = generate_resize_matrix(scale, inv=inv)
    rot_mat = generate_rotation_matrix(angle, input_shape, inv=inv)

    if inv:
        transform = crop_mat @ resize_mat @ rot_mat
    else:
        transform = rot_mat @ resize_mat @ crop_mat

    transform = tf.reshape(transform, [-1])[:-1]
    return transform


def generate_crop_matrix(bbox_center, scale, input_shape, inv: bool = False):
    crop_x = bbox_center[0] - (input_shape[1] * scale) / 2
    crop_y = bbox_center[1] - (input_shape[0] * scale) / 2

    crop_mat = tf.cond(
        tf.math.equal(inv, True),
        lambda: tf.reshape(
            [
                1., 0., crop_x,
                0., 1., crop_y,
                0., 0., 1.
            ],
            shape=[3, 3]
        ),
        lambda: tf.reshape(
            [
                1., 0., -crop_x,
                0., 1., -crop_y,
                0., 0., 1.
            ],
            shape=[3, 3]
        )
    )
    return crop_mat


def generate_resize_matrix(scale, inv: bool = False):
    resize_mat = tf.cond(
        tf.math.equal(inv, True),
        lambda: tf.reshape(
            [
                scale, 0., 0.,
                0., scale, 0.,
                0., 0., 1.
            ],
            shape=[3, 3]
        ),
        lambda: tf.reshape(
            [
                1. / scale, 0., 0.,
                0., 1. / scale, 0.,
                0., 0., 1.
            ],
            shape=[3, 3]
        ),
    )
    return resize_mat


def generate_rotation_matrix(angle, input_shape, inv: bool = False):
    radian = angle / 180 * tf.constant(math.pi, dtype=tf.float32)

    # move center to origin
    # 이미지 중심을 원점으로 이동
    translation1 = tf.reshape(
        tf.convert_to_tensor(
            [
                1., 0., (input_shape[1] / 2),
                0., 1., (input_shape[0] / 2),
                0., 0., 1.
            ],
            dtype=tf.float32
        ),
        shape=[3, 3]
    )

    # move back to center
    # 다시 이미지 중심으로 이동
    translation2 = tf.reshape(
        tf.convert_to_tensor(
            [
                1., 0., -(input_shape[1] / 2),
                0., 1., -(input_shape[0] / 2),
                0., 0., 1.
            ],
            dtype=tf.float32
        ),
        shape=[3, 3]
    )

    rotation_mat = tf.cond(
        tf.math.equal(inv, True),
        lambda: tf.reshape(
            tf.convert_to_tensor(
                [
                    tf.math.cos(radian), tf.math.sin(radian), 0.,
                    -tf.math.sin(radian), tf.math.cos(radian), 0.,
                    0., 0., 1.
                ],
                dtype=tf.float32
            ),
            shape=[3, 3]
        ),
        lambda: tf.reshape(
            tf.convert_to_tensor(
                [
                    tf.math.cos(radian), -tf.math.sin(radian), 0.,
                    tf.math.sin(radian), tf.math.cos(radian), 0.,
                    0., 0., 1.
                ],
                dtype=tf.float32
            ),
            shape=[3, 3]
        )
    )
    return translation1 @ rotation_mat @ translation2


def preprocess(
    img, bbox, kp,
    use_image_norm: bool = True,
    means: List = [0.485, 0.456, 0.406],
    stds: List = [0.229, 0.224, 0.225],
    scale_factor: float = 0.3,
    rotation_prob: float = 0.6,
    rotation_factor: int = 40,
    flip_prob: float = 0.5,
    flip_kp_indices: List = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    half_body_prob: float = 1.0,
    half_body_min_kp: int = 8,
    kpt_upper: List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    input_shape: List = [192, 192, 3],
    album: bool = False,
    use_aug: bool = True,
):
    kp = tf.cast(kp, tf.float32)

    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = tf.cast([x1, y1, x1 + w, y1 + h], tf.float32)
    bbox_center = tf.cast([x1 + w / 2., y1 + h / 2.], tf.float32)

    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]  # scale with bbox
    angle = 0.

    # augmentation
    joint_vis = tf.math.reduce_sum(tf.cast(kp[:, 2] > 0, tf.int32))
    bbox_center, scale = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.greater(joint_vis, half_body_min_kp)
        & tf.math.less(tf.random.uniform([]), half_body_prob),
        lambda: half_body_transform(
            kp, bbox_center, scale, kpt_upper, input_shape
        ),
        lambda: (bbox_center, scale)
    )
    # 1. scale
    scale *= tf.cond(
        tf.math.equal(use_aug, True),
        lambda: tf.clip_by_value(tf.random.normal([]) * scale_factor + 1,
                                 1 - scale_factor,
                                 1 + scale_factor),
        lambda: 1.0
    )
    # 2. rotation
    angle = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.less_equal(tf.random.uniform([]), rotation_prob),
        lambda: tf.clip_by_value(
            tf.random.normal([]) * rotation_factor,
            -2 * rotation_factor,
            2 * rotation_factor
        ),
        lambda: angle
    )
    # 3. horizontal flip
    img, bbox_center, kp = tf.cond(
        tf.math.equal(use_aug, True)
        & tf.math.less_equal(tf.random.uniform([]), flip_prob),
        lambda: horizontal_flip(img, bbox_center, kp, flip_kp_indices),
        lambda: (img, bbox_center, kp)
    )
    # transform to the object's center
    img, M = affine_transform(img, bbox_center, angle, scale, input_shape[:2])

    xy = kp[:, :2]
    xy = tf.transpose(tf.matmul(M[:, :2], xy, transpose_b=True)) + M[:, -1]

    # adjust visibility if coordinates are outside crop
    vis = tf.cast(kp[:, 2] > 0, tf.float32)  # vis==2인 경우 처리하기 위함
    vis *= tf.cast(
        (
            (xy[:, 0] >= 0) &
            (xy[:, 0] < input_shape[1]) &
            (xy[:, 1] >= 0) &
            (xy[:, 1] < input_shape[0])
        ),
        tf.float32
    )
    kp = tf.concat([xy, tf.expand_dims(vis, axis=1)], axis=1)

    img = tf.cond(
        tf.math.logical_and(use_aug, album),
        lambda: apply_albumentaion(img),
        lambda: img
    )
    img = tf.cast(img, tf.float32)
    img = tf.cond(
        tf.math.equal(use_image_norm, True),
        lambda: normalize_image(img, means, stds),
        lambda: img
    )
    target = generate_target(kp, input_shape)
    return img, target


def normalize_image(image, means: List, stds: List):
    image /= 255.
    image -= [[means]]
    image /= [[stds]]
    return image


def get_center_keypoints(keypoints, vis):
    """keypoint: [K, 2]"""
    # exclude the invisible keypoints
    vis = tf.math.greater(vis, 0)
    kpt_idx = tf.where(vis)
    keypoints = tf.gather_nd(keypoints, kpt_idx)
    return tf.math.reduce_mean(keypoints, axis=0)


def half_body_transform(
    joints,
    center,
    scale: float,
    kpt_upper: List,
    input_shape: List
):
    K = tf.shape(joints)[0]
    num_upper = tf.shape(kpt_upper)[0]
    vis_mask = joints[:, 2] > 0
    kpt_upper = tf.reshape(kpt_upper, (-1, 1))
    upper_body_mask = tf.scatter_nd(kpt_upper, tf.ones(num_upper,), shape=(K,))
    upper_body_mask = tf.cast(upper_body_mask, tf.bool)
    lower_body_mask = tf.math.logical_not(upper_body_mask)
    lower_body_mask = tf.math.logical_and(lower_body_mask, vis_mask)
    upper_body_mask = tf.math.logical_and(upper_body_mask, vis_mask)
    upper = tf.boolean_mask(joints, upper_body_mask)
    lower = tf.boolean_mask(joints, lower_body_mask)

    selected_joints = tf.cond(
        tf.math.less(tf.random.uniform([]), 0.5)
        & tf.math.greater(tf.shape(upper)[0], 2),
        lambda: upper,
        lambda: lower
    )
    center, scale = tf.cond(
        tf.math.greater_equal(tf.shape(selected_joints)[0], 2),
        lambda: get_center_scale_from_half_body(
            selected_joints, input_shape
        ),
        lambda: (center, scale)
    )
    return center, scale


def get_center_scale_from_half_body(
    selected_joints,
    input_shape: List
):
    center = tf.math.reduce_mean(selected_joints[:, :2], axis=0)
    left_top = tf.math.reduce_min(selected_joints[:, :2], axis=0)
    right_bottom = tf.math.reduce_max(selected_joints[:, :2], axis=0)
    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]
    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]
    scale = scale * 1.5
    scale = h / input_shape[0]
    return center, scale


def horizontal_flip(
    img,
    center,
    kp,
    flip_kp_indices: List
):
    img_w = tf.cast(tf.shape(img)[1], tf.float32)
    img = img[:, ::-1, :]
    center_x = img_w - 1 - center[0]
    kp_x = img_w - 1 - kp[:, 0]
    kp = tf.concat([tf.expand_dims(kp_x, axis=1), kp[:, 1:]], axis=-1)
    kp = tf.gather(kp, flip_kp_indices, axis=0)
    center = tf.cast([center_x, center[1]], tf.float32)
    return img, center, kp


def generate_target(keypoints, input_shape: List):
    target_visible = keypoints[:, -1]

    scale = tf.convert_to_tensor(
        [input_shape[1], input_shape[0]],
        dtype=keypoints.dtype
    )
    target = keypoints[:, :2] / scale - 0.5

    # masking values not existing in [-0.5 ~ 0.5]
    target_visible *= tf.cast((
        (target[:, 0] <= 0.5) &
        (target[:, 0] >= -0.5) &
        (target[:, 1] <= 0.5) &
        (target[:, 1] >= -0.5)), tf.float32)

    target = tf.concat(
        [target, tf.expand_dims(target_visible, axis=1)],
        axis=-1
    )
    return target
