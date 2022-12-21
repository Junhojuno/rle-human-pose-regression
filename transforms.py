import math
from typing import List, Tuple
import tensorflow as tf
import tensorflow_addons as tfa


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


def transform(img, scale, angle, center, output_shape) -> Tuple[tf.Tensor, tf.Tensor]:
    tx = center[0] - output_shape[1] * scale / 2
    ty = center[1] - output_shape[0] * scale / 2
    
    # for offsetting translations caused by rotation:
    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    rx = (1 - tf.cos(angle)) * output_shape[1] * scale / 2 - tf.sin(angle) * output_shape[0] * scale / 2
    ry = tf.sin(angle) * output_shape[1] * scale / 2 + (1 - tf.cos(angle)) * output_shape[0] * scale / 2

    transform = [scale * tf.cos(angle), scale * tf.sin(angle), rx + tx,
                 -scale * tf.sin(angle), scale * tf.cos(angle), ry + ty,
                 0., 0.]
    
    img = tfa.image.transform(tf.expand_dims(img, axis=0),
                              tf.expand_dims(transform, axis=0),
                              fill_mode='constant',
                              output_shape=output_shape[:2])
    img = tf.squeeze(img)
    
    # transform for keypoints
    alpha = 1 / scale * tf.cos(-angle)
    beta = 1 / scale * tf.sin(-angle)
    
    rx_xy = (1 - alpha) * center[0] - beta * center[1]
    ry_xy = beta * center[0] + (1 - alpha) * center[1]
    
    transform_xy = [[alpha, beta],
                    [-beta, alpha]]
    
    tx_xy = center[0] - output_shape[1] / 2
    ty_xy = center[1] - output_shape[0] / 2
    
    M = tf.concat([transform_xy, [[rx_xy - tx_xy], [ry_xy - ty_xy]]], axis=1)
    return img, M


def preprocess(
    img, bbox, kp,
    use_image_norm: bool = True,
    means: List = [0.485, 0.456, 0.406],
    stds: List = [0.229, 0.224, 0.225],
    scale_factor: float = 0.3,
    rotation_prob: float = 0.6,
    rotation_factor: int = 40,
    flip_prob: float = 0.5,
    flip_kp_indices: List = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    half_body_prob: float = 1.0,
    half_body_min_kp: int = 8,
    kpt_upper: List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    input_shape: List = [192, 192, 3],
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
    scale = (h * 1.25) / input_shape[0] # scale with bbox
    angle = 0.
    
    # augmentation
    joint_vis = tf.math.reduce_sum(tf.cast(kp[:, 2] > 0, tf.int32))
    bbox_center, scale = tf.cond(
        tf.math.equal(use_aug, True) & tf.math.greater(joint_vis, half_body_min_kp) & tf.math.less(tf.random.uniform([]), half_body_prob),
        lambda: half_body_transform(kp, bbox_center, scale, kpt_upper, input_shape),
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
        tf.math.equal(use_aug, True) & tf.math.less_equal(tf.random.uniform([]), rotation_prob),
        lambda: tf.clip_by_value(tf.random.normal([]) * rotation_factor,
                                 -2 * rotation_factor,
                                 2 * rotation_factor) / 180 * tf.constant(math.pi, dtype=tf.float32),
        lambda: angle
    )
    # 3. horizontal flip
    img, bbox_center, kp = tf.cond(
        tf.math.equal(use_aug, True) & tf.math.less_equal(tf.random.uniform([]), flip_prob),
        lambda: horizontal_flip(img, bbox_center, kp, flip_kp_indices),
        lambda: (img, bbox_center, kp)
    )
    # transform to the object's center
    img, M = transform(img, scale, angle, bbox_center, input_shape[:2])
    
    xy = kp[:, :2]
    xy = tf.transpose(tf.matmul(M[:, :2], xy, transpose_b=True)) + M[:, -1]
    
    # adjust visibility if coordinates are outside crop
    vis = tf.cast(kp[:, 2] > 0, tf.float32) # vis==2인 경우 처리하기 위함
    vis *= tf.cast((
            (xy[:, 0] >= 0) &
            (xy[:, 0] < input_shape[1]) &
            (xy[:, 1] >= 0) &
            (xy[:, 1] < input_shape[0])), tf.float32)    
    kp = tf.concat([xy, tf.expand_dims(vis, axis=1)], axis=1)

    img = tf.cast(img, tf.float32)
    img = tf.cond(
        tf.math.equal(use_image_norm, True),
        lambda: normalize_image(img, means, stds),
        lambda: img
    )
    
    # target, target_weight, _ = generate_target(kp, input_shape)
    # return img, target, target_weight
    target = generate_target(kp, input_shape)
    return img, target


def normalize_image(image, means, stds):
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


def half_body_transform(joints, center, scale, kpt_upper, input_shape):
    K = tf.shape(joints)[0]
    num_upper = tf.shape(kpt_upper)[0]
    vis_mask = joints[:,2] > 0
    kpt_upper = tf.reshape(kpt_upper, (-1, 1))
    upper_body_mask = tf.scatter_nd(kpt_upper, tf.ones(num_upper,), shape=(K,))
    upper_body_mask = tf.cast(upper_body_mask, tf.bool)
    lower_body_mask = tf.math.logical_not(upper_body_mask)
    lower_body_mask = tf.math.logical_and(lower_body_mask, vis_mask)
    upper_body_mask = tf.math.logical_and(upper_body_mask, vis_mask)
    upper = tf.boolean_mask(joints, upper_body_mask)
    lower = tf.boolean_mask(joints, lower_body_mask)
    
    selected_joints = tf.cond(
        tf.math.less(tf.random.uniform([]), 0.5) & tf.math.greater(tf.shape(upper)[0], 2),
        lambda: upper,
        lambda: lower
    )
    center, scale = tf.cond(
        tf.math.greater_equal(tf.shape(selected_joints)[0], 2),
        lambda: _half_body_transform(selected_joints, input_shape),
        lambda: (center, scale)
    )
    return center, scale


def _half_body_transform(selected_joints, input_shape):
    center = tf.math.reduce_mean(selected_joints[:, :2], axis = 0)
    left_top = tf.math.reduce_min(selected_joints[:, :2], axis = 0)
    right_bottom = tf.math.reduce_max(selected_joints[:, :2], axis = 0)
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


def horizontal_flip(img, center, kp, flip_kp_indices):
    img_w = tf.cast(tf.shape(img)[1], tf.float32)
    img = img[:, ::-1, :]
    center_x = img_w - 1 - center[0]
    kp_x = img_w - 1 - kp[:, 0]
    kp = tf.concat([tf.expand_dims(kp_x, axis=1), kp[:, 1:]], axis=-1)
    kp = tf.gather(kp, flip_kp_indices, axis=0)
    center = tf.cast([center_x, center[1]], tf.float32) 
    return img, center, kp


def generate_target(keypoints, input_shape):
    target_visible = keypoints[:, -1]
    
    scale = tf.convert_to_tensor([input_shape[1], input_shape[0]], keypoints.dtype)
    target = keypoints[:, :2] / scale - 0.5
    
    # masking values not existing in [-0.5 ~ 0.5]
    target_visible *= tf.cast((
            (target[:, 0] <= 0.5) &
            (target[:, 0] >= -0.5) &
            (target[:, 1] <= 0.5) &
            (target[:, 1] >= -0.5)), tf.float32)
    
    target = tf.concat([target, tf.expand_dims(target_visible, axis=1)], axis=-1)
    return target


# def generate_target(keypoints, input_shape):
#     target_visible = keypoints[:, 2:]
#     target_weight = tf.concat([target_visible, target_visible], axis=-1)
    
#     scale = tf.convert_to_tensor([input_shape[1], input_shape[0]], keypoints.dtype)
#     target = keypoints[:, :2] / scale - 0.5
    
#     # masking values not existing in [-0.5 ~ 0.5]
#     target_visible *= tf.cast((
#             (target[:, 0] <= 0.5) &
#             (target[:, 0] >= -0.5) &
#             (target[:, 1] <= 0.5) &
#             (target[:, 1] >= -0.5)), tf.float32)
    
#     target = tf.reshape(target, [-1,])
#     target_weight = tf.reshape(target_weight, [-1,])
#     return target, target_weight, target_visible
