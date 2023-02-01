import tensorflow as tf


def calc_coord_accuracy(y_true, y_pred, heatmap_shape, thr: int = 0.5):
    pred_mu = y_pred.mu
    scale = tf.convert_to_tensor(
        [heatmap_shape[1], heatmap_shape[0]],
        pred_mu.dtype
    )
    pred_mu = (pred_mu + 0.5) * scale

    gt_mu, gt_mask = tf.split(y_true, (2, 1), axis=-1)
    gt_mu = (gt_mu + 0.5) * scale

    pred_mu = pred_mu * gt_mask
    gt_mu = gt_mu * gt_mask

    # calculate distance
    norm = tf.ones([tf.shape(gt_mu)[0], 1, 2], dtype=scale.dtype) * scale / 10.
    dists, valid_mask, n_valids = cal_dist(gt_mu, pred_mu, norm)
    n_corrects = tf.math.reduce_sum(
        tf.cast(dists < thr, tf.float32) * valid_mask
    )
    acc = tf.math.divide_no_nan(n_corrects, n_valids)
    return acc


def cal_dist(target, pred, norm):
    valid_mask = tf.math.reduce_all(target > 1, axis=-1)
    dists = tf.math.reduce_euclidean_norm(
        (pred / norm) - (target / norm),
        axis=-1
    )  # (B, K)
    dists = tf.where(valid_mask, dists, -tf.ones_like(dists))
    valid_mask = tf.cast(valid_mask, tf.float32)
    n_valids = tf.math.reduce_sum(valid_mask)
    return dists, valid_mask, n_valids
