import math
import tensorflow as tf
from tensorflow import keras
from keras import losses


class RLELoss(losses.Loss):

    def __init__(self, reduction=losses.Reduction.NONE, name=None):
        super().__init__(reduction, name)
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return tf.math.log(sigma / self.amp) + tf.math.abs(gt_uv - pred_jts) / (tf.math.sqrt(2) * sigma + 1e-9)
    
    def call(self, y_true, y_pred):
        mu_hat, sigma_hat = tf.split(y_pred, (2, 1), axis=-1)
        return
