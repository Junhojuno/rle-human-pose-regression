import math
import tensorflow as tf
from tensorflow.keras import losses


class RLELoss(losses.Loss):
    """Residual Likelihood Estimation Loss function

    L = -logP(bar_mu)
      = -logQ(bar_mu) - logG(bar_mu) - log(s) + log(sigma_hat)
      = -logQ(bar_mu) - logG(bar_mu) + log(sigma_hat) # remove constant in loss
    """

    def __init__(self, reduction=losses.Reduction.NONE, name='RLELoss'):
        super().__init__(reduction, name)
        self.amp = 1 / math.sqrt(2 * math.pi)

    def minus_logQ(self, gt_uv, pred_jts, sigma):
        """minus Log Laplace

        It means -logQ(bar_mu) in the papaer.
        And Q has followed Laplace distribution.
        """
        return tf.math.log(sigma / self.amp) + tf.math.abs(gt_uv - pred_jts) / (tf.math.sqrt(2.) * sigma + 1e-9)

    def call(self, y_true, y_pred):
        gt_mu, gt_visibility = tf.split(y_true, (2, 1), axis=-1)
        pred_mu, pred_sigma = y_pred.mu, y_pred.sigma
        nf_loss = y_pred.nf_loss

        nf_loss = nf_loss * gt_visibility  # -logG + log(sigma)

        minus_Q_logprob = self.minus_logQ(gt_mu, pred_mu, pred_sigma)
        minus_Q_logprob *= gt_visibility
        loss = nf_loss + minus_Q_logprob
        return tf.math.reduce_mean(loss, axis=(1, 2))  # (B,)
