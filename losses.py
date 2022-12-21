import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import losses

from model import RealNVP, nets, nett


prior = tfp.distributions.MultivariateNormalFullCovariance(tf.zeros(2), tf.eye(2))
masks = tf.tile(
    tf.convert_to_tensor([[0, 1], [1, 0]], dtype=tf.float32),
    multiples=[3, 1]
)


class RLELoss(losses.Loss):

    def __init__(self, reduction=losses.Reduction.NONE, name=None):
        super().__init__(reduction, name)
        self.amp = 1 / math.sqrt(2 * math.pi)
        
        masks = tf.tile(
            tf.convert_to_tensor([[0, 1], [1, 0]], dtype=tf.float32),
            multiples=[3, 1]
        )
        prior = tfp.distributions.MultivariateNormalFullCovariance(tf.zeros(2), tf.eye(2))
        self.flow_model = RealNVP(nets, nett, masks, prior)

    def logQ(self, gt_uv, pred_jts, sigma):
        return tf.math.log(sigma / self.amp) + tf.math.abs(gt_uv - pred_jts) / (tf.math.sqrt(2.) * sigma + 1e-9)
    
    def call(self, y_true, y_pred):
        K = tf.shape(y_pred)[1]
        gt_mu, gt_visibility = tf.split(y_true, (2, 1), axis=-1)
        pred_mu, pred_sigma = tf.split(y_pred, (2, 2), axis=-1)
        
        bar_mu = (pred_mu - gt_mu) / pred_sigma
        log_phi = self.flow_model(tf.reshape(bar_mu, [-1, 2]))
        log_phi = tf.reshape(log_phi, [-1, K, 1])
        nf_loss = tf.math.log(pred_sigma) - log_phi
        
        # gt_mu_weight = tf.reshape(target_weight, tf.shape(pred_mu)) # (B, K, 2)
        nf_loss = nf_loss * gt_visibility
    
        Q_logprob = self.logQ(gt_mu, pred_mu, pred_sigma) * gt_visibility
        loss = nf_loss + Q_logprob
        return tf.math.reduce_mean(loss, axis=(1, 2)) # (B,)


def rle_loss(target, target_weight, pred):
    pred_mu, pred_sigma = tf.split(pred, 2, axis=-1)
    K = tf.shape(pred_mu)[1]
    bar_mu = (pred_mu - target) / pred_sigma
    log_phi = RealNVP(nets, nett, masks, prior)(tf.reshape(bar_mu, [-1, 2]))
    log_phi = tf.reshape(log_phi, [-1, K, 1])
    
    nf_loss = tf.math.log(pred_sigma) - log_phi
    gt_mu = tf.reshape(target, tf.shape(pred_mu))
    gt_mu_weight = tf.reshape(target_weight, tf.shape(pred_mu)) # (B, K, 2)
    
    nf_loss = nf_loss * gt_mu_weight[..., :1]
    
    Q_logprob = cal_log_Q(gt_mu, pred_mu, pred_sigma) * gt_mu_weight
    loss = nf_loss + Q_logprob
    return tf.math.reduce_mean(loss, axis=(1, 2)) # (B,)


def cal_log_Q(gt_mu, pred_mu, pred_sigma):
    amp = 1 / math.sqrt(2 * math.pi)
    return tf.math.log(pred_sigma / amp) + tf.math.abs(gt_mu - pred_mu) / (tf.math.sqrt(2) * pred_sigma + 1e-9)
