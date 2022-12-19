import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import Model, Sequential
from keras import layers
from keras.applications.resnet import ResNet50


def nets(units=64):
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


def nett(units=64):
    return Sequential(
        [
            layers.Dense(units),
            layers.LeakyReLU(),
            layers.Dense(units),
            layers.LeakyReLU(),
            layers.Dense(2),
        ]
    )


class XavierUniform(keras.initializers.GlorotUniform):

    def __init__(self, scale=0.01, seed=None):
        super().__init__(scale=scale, seed=seed)

    def get_config(self):
        return {
            "scale": self.scale,
            "seed": self.seed
        }
    

class Linear(layers.Layer):
    """layer for mu & sigma of kpt coordinates

    normalize 사용 & kernel, bias 따로 계산하는 특징이 있음
    """

    def __init__(self, units, use_bias=True, use_norm=True, name='linear'):
        super().__init__(name=name)
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.dense = layers.Dense(
            units, use_bias=use_bias, kernel_initializer=XavierUniform()
        )

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
            bound = 1 / tf.math.sqrt(last_dim)
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=keras.initializers.RandomUniform(-bound, bound),
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


class RegressFlow(Model):

    def __init__(
        self,
        num_keypoints=17,
        input_shape=[256, 192, 3],
        sigmoid_fn=layers.Activation('sigmoid'),
        is_training=False
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sigmoid_fn = sigmoid_fn
        self.is_training = is_training
        
        self.backbone = ResNet50(include_top=False, input_shape=input_shape)
        self.gap = layers.GlobalAveragePooling2D()
        self.dense_kpt_mu = Linear(num_keypoints * 2)
        self.dense_kpt_sigma = Linear(num_keypoints * 2, use_norm=False)

        masks = tf.tile(
            tf.convert_to_tensor([[0, 1], [1, 0]], dtype=tf.float32),
            multiples=[3, 1]
        )
        prior = tfp.distributions.MultivariateNormalFullCovariance(tf.zeros(2), tf.eye(2))
        self.flow_model = RealNVP(nets, nett, masks, prior)

    def call(self, inputs, mu_g=None):
        feat = self.backbone(inputs)
        feat = self.gap(feat)
        
        mu_hat = layers.Reshape(
            [inputs.shape[0], self.num_keypoints, 2]
        )(self.dense_kpt_mu(feat))
        sigma_hat = layers.Reshape(
            [inputs.shape[0], self.num_keypoints, 2]
        )(self.dense_kpt_sigma(feat))
        sigma_hat = self.sigmoid_fn(sigma_hat)
        
        # if self.is_training and mu_g is not None:
        #     bar_mu = (mu_hat - mu_g) / sigma_hat
        #     log_phi = self.flow_model(tf.reshape(bar_mu, [-1, 2]))
        #     log_phi = tf.reshape(log_phi, [-1, self.num_keypoints, 1])

        if self.is_training:
            return layers.Concatenate()([mu_hat, sigma_hat])
        score = tf.math.reduce_mean(1 - sigma_hat, axis=-1, keepdims=True)
        return layers.Concatenate()([mu_hat, score])
            
            

class RealNVP(Model):
    """Density Estimation using Real NVP

    models using Real-valued Non-Volume Preserving transformations,
    which are powerful, stably invertible, and learnable
    https://arxiv.org/abs/1605.08803
    """
    
    def __init__(self, nets, nett, mask, prior):
        super().__init__()
        self.prior = prior
        self.t = [nett() for _ in range(len(mask))] # translation
        self.s = [nets() for _ in range(len(mask))] # scale

    def f(self, x):
        """
        transformation of
            x(image) -> z(latent variable)
        """
        pass

    def g(self, z):
        """
        inverse transformation of
            z(latent variable) -> x(image)
        """
        pass

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tf.math.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J = tf.zeros_like(x, dtype=x.dtype)
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
        return self.log_prob(x)
