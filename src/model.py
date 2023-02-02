"""RLE model"""
from typing import List, Optional, Callable
from easydict import EasyDict
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers

from src.backbone import build_backbone


def nets(units: Optional[int] = 64):
    return Sequential(
        [
            layers.Dense(units),
            layers.LeakyReLU(),
            layers.Dense(units),
            layers.LeakyReLU(),
            layers.Dense(2),
            layers.Activation('tanh')
        ]
    )


def nett(units: Optional[int] = 64):
    return Sequential(
        [
            layers.Dense(units),
            layers.LeakyReLU(),
            layers.Dense(units),
            layers.LeakyReLU(),
            layers.Dense(2),
        ]
    )


class XavierUniform(tf.keras.initializers.GlorotUniform):

    def __init__(
        self,
        scale: Optional[float] = 0.01,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.scale = scale
        self.seed = seed

    def get_config(self):
        return {
            "scale": self.scale,
            "seed": self.seed
        }


class Linear(layers.Layer):
    """layer for mu & sigma of kpt coordinates

    normalize 사용 & kernel, bias 따로 계산하는 특징이 있음
    """

    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        use_norm: bool = True,
        name: str = 'linear'
    ):
        super().__init__(name=name)
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.dense = layers.Dense(
            units, use_bias=use_bias, kernel_initializer=XavierUniform()
        )
        self.units = units

    def build(self, input_shape):
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=XavierUniform(),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            bound = 1 / tf.math.sqrt(tf.cast(last_dim, tf.float32))
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=tf.keras.initializers.RandomUniform(-bound, bound),
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=True,
            )

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.use_norm:
            norm = tf.norm(inputs, axis=-1, keepdims=True)
            output /= norm
        if self.use_bias:
            output += self.bias
        return output


class RealNVP(Model):
    """Density Estimation using Real NVP

    models using Real-valued Non-Volume Preserving transformations,
    which are powerful, stably invertible, and learnable
    https://arxiv.org/abs/1605.08803
    """

    def __init__(
        self,
        nets: Callable,
        nett: Callable,
        mask: tf.Tensor,
        prior: tfp.distributions.MultivariateNormalDiag
    ):
        super().__init__()
        self.prior = prior
        self.t = [nett() for _ in range(len(mask))]  # translation
        self.s = [nets() for _ in range(len(mask))]  # scale
        self.mask = mask

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tf.math.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J = tf.zeros_like(x[..., 0], dtype=x.dtype)
        z = x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * tf.math.exp(-s) + z_
            log_det_J -= tf.math.reduce_sum(s, axis=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def call(self, x):
        return self.loOpt


class RLEModel(Model):
    """Regression Model with Residual Log-likelihood"""

    def __init__(
        self,
        num_keypoints: int = 17,
        input_shape: List = [256, 192, 3],
        backbone_type: str = 'resnet50',
        sigmoid_fn: Callable = layers.Activation('sigmoid'),
        is_training: bool = False
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sigmoid_fn = sigmoid_fn

        self.backbone = build_backbone(backbone_type, input_shape)
        self.gap = layers.GlobalAveragePooling2D()
        self.dense_kpt_mu = Linear(num_keypoints * 2)
        self.dense_kpt_sigma = Linear(num_keypoints * 2, use_norm=False)
        # self.dense_kpt_mu = layers.Dense(num_keypoints * 2)
        # self.dense_kpt_sigma = layers.Dense(num_keypoints * 2)

        if is_training:
            masks = tf.tile(
                tf.convert_to_tensor([[0, 1], [1, 0]], dtype=tf.float32),
                multiples=[3, 1]
            )
            prior = tfp.distributions.MultivariateNormalDiag(
                tf.zeros(2), tf.ones(2)
            )
            self.flow_model = RealNVP(nets, nett, masks, prior)
            self.flow_model.build([None, 2])

        self.is_training = is_training

    def call(self, inputs, mu_g: Optional[tf.Tensor] = None) -> EasyDict:
        feat = self.backbone(inputs)
        feat = self.gap(feat)

        # heads
        mu_hat = layers.Reshape(
            [self.num_keypoints, 2]
        )(self.dense_kpt_mu(feat))
        sigma_hat = layers.Reshape(
            [self.num_keypoints, 2]
        )(self.dense_kpt_sigma(feat))
        sigma_hat = self.sigmoid_fn(sigma_hat)

        scores = 1 - sigma_hat
        scores = tf.math.reduce_mean(scores, -1, keepdims=True)

        if self.is_training and mu_g is not None:
            # log_phi means logG_phi(bar_mu) in the paper.
            bar_mu = (mu_hat - mu_g) / sigma_hat
            log_phi = self.flow_model(tf.reshape(bar_mu, [-1, 2]))
            log_phi = tf.reshape(log_phi, [-1, self.num_keypoints, 1])

            nf_loss = tf.math.log(sigma_hat) - log_phi
        else:
            nf_loss = None

        output = EasyDict(
            mu=mu_hat,
            sigma=sigma_hat,
            maxvals=tf.cast(scores, tf.float32),
            nf_loss=nf_loss
        )
        return output
