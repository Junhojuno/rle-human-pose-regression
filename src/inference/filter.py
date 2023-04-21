"""
ref: https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/movenet/detector.ts
"""
import math
import numpy as np

MICRO_SECONDS_TO_SECOND = 1e-6
SECOND_TO_MICRO_SECONDS = 1e+6


class LowPassFilter:
    """
    for exponential smoothing

    reference
    https://github.com/tensorflow/tfjs-models/blob/70848f52338d33c3db6b491b5791b25bd6bcb83f/shared/filters/low_pass_filter.ts#L27
    """
    initialized = False
    raw_value = 0.0
    stored_value = 0.0

    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def apply(self, value, threshold=None):
        if self.initialized:
            if threshold is None:
                """
                Regular low-pass filter
                    - result = alpha * X_i + (1 - alpha) * stored_value;
                    - We need to reformat the formula to be able to conveniently apply
                    - another optional non-linear function to the
                    - (value - self.stored_value) part.
                """
                result = self.stored_value\
                    + self.alpha * (value - self.stored_value)
            else:
                """
                another optional non-linear function: asinh(Inverse Hyperbolic Sin function)
                    - Add additional non-linearity to cap extreme value.
                    - More specifically,
                    - assume x = (value - self.stored_value), 
                    - when x is close zero, the derived x is close to x.
                    - when x is several magnitudes larger, the derived x grows much slower then x.
                    - It behaves like sign(x)log(abs(x)).
                """
                result = self.stored_value \
                    + self.alpha * threshold \
                    * math.asinh((value - self.stored_value) / threshold)
        else:
            result = value
            self.initialized = True
        self.raw_value = value
        self.stored_value = result
        return result

    def apply_with_alpha(self, value, alpha, threshold=None):
        self.alpha = alpha
        return self.apply(value, threshold)

    def has_last_raw_value(self):
        return self.initialized

    def get_last_raw_value(self):
        return self.raw_value

    def reset(self):
        self.initialized = False


class OneEuroFilter:
    """reference:
    https://github.com/tensorflow/tfjs-models/blob/70848f5233/shared/filters/one_euro_filter.ts
    """

    def __init__(
        self,
        frequency: int,
        min_cutoff: float,
        beta: int,
        cutoff_threshold: float,
        beta_threshold: float,
        cutoff_derivative: float
    ) -> None:
        """
        Args:
            frequency (int): frame per second. default=30
            min_cutoff (float): reasonable middle-ground value. keypoint 떨림(jitter)현상이 있을 때, 값을 낮추자!
            beta (int): speed coefficient. 빠른 동작에 keypoint가 잘 따라가지 못한다면 값을 키우자!
            cutoff_threshold (float): smoothing할 때, non-linear function asinh에 곱하는 threshold를 구할 때 사용됨
            beta_threshold (float): smoothing할 때, non-linear function asinh에 곱하는 threshold를 구할 때 사용됨
            cutoff_derivative (float): smoothing된 변화율을 구할 때, 활용할 alpha를 계산할 때 사용할 cutoff
        """
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.cutoff_threshold = cutoff_threshold
        self.beta_threshold = beta_threshold
        self.cutoff_derivative = cutoff_derivative
        self.x = LowPassFilter(self.get_alpha(min_cutoff))
        self.dx = LowPassFilter(self.get_alpha(cutoff_derivative))
        self.last_timestamp = 0

    def apply(self, value=None, ms=None, scale=None):
        """하나의 keypoint x 또는 y 좌표값(value)에 filter를 적용하는 과정"""
        if value is None:
            return value
        ms = int(ms)
        if self.last_timestamp >= ms:
            # Results are unpreditable in this case,
            # so nothing to do but return same value.
            return value
        if self.last_timestamp != 0 and ms != 0:
            # Update the sampling frequency based on timestamps.
            self.frequency = \
                1 / ((ms - self.last_timestamp) * MICRO_SECONDS_TO_SECOND)
        self.last_timestamp = ms

        """
        1. derivative of the signal
        """
        d_value = (value - self.x.get_last_raw_value())\
            * scale\
            * self.frequency if self.x.has_last_raw_value() else 0
        """
        2. exponential smoothing to the derivative
        """
        ed_value = self.dx.apply_with_alpha(
            d_value,
            self.get_alpha(self.cutoff_derivative)
        )
        """
        3. renew cutoff frequency
        """
        cutoff = self.min_cutoff + self.beta * abs(ed_value)
        """
        4. threshold for non-linear smoothing/filter
        보통의 low-pass filter가 아닌 non-linear한 형태의 filter를 적용하는 데 사용되는 threshold
        """
        threshold = self.cutoff_threshold + self.beta_threshold * abs(ed_value) \
            if self.cutoff_threshold is not None else None
        return self.x.apply_with_alpha(
            value, self.get_alpha(cutoff), threshold
        )

    def get_alpha(self, cutoff):
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * math.pi * cutoff)
        result = 1.0 / (1.0 + (tau / te))
        return result


class KeypointOneEuroFilter:
    """
    this module is heavily borrowed from below
    https://github.com/tensorflow/tfjs-models/blob/70848f52338d33c3db6b491b5791b25bd6bcb83f/shared/filters/keypoints_one_euro_filter.ts#L31

    """
    x_filter = None
    y_filter = None

    def __init__(
        self,
        frequency: int = 30,
        min_cutoff: float = 2.5,
        beta: float = 300.0,
        cutoff_derivative: float = 2.5,
        cutoff_threshold: float = 0.5,
        beta_threshold: float = 5.0,
        disable_value_scaling: bool = True
    ) -> None:
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.cutoff_derivative = cutoff_derivative
        self.cutoff_threshold = cutoff_threshold
        self.beta_threshold = beta_threshold
        self.disable_value_scaling = disable_value_scaling

    def apply(self, keypoints, ms, object_scale):
        if keypoints is None:
            self.reset()
            return None

        # Initialize filters once.
        self.initialize(keypoints)

        # Get value scale as inverse value of the object scale.
        # If value is too small smoothing will be disabled
        # and keypoints will be returned as is.
        value_scale = 1
        if not self.disable_value_scaling:
            value_scale = 1.0 / object_scale

        out_keypoints = np.array(
            [
                [
                    self.x_filter[i].apply(kpt[0], ms, value_scale),
                    self.y_filter[i].apply(kpt[1], ms, value_scale)
                ]
                for i, kpt in enumerate(keypoints)
            ]
        )  # (17, 2). keypoint x, y
        out_keypoints = np.concatenate(
            [out_keypoints, keypoints[:, 2:]], axis=-1)
        return out_keypoints

    def initialize(self, keypoints):
        """Initialize filters once."""
        if self.x_filter is None or len(self.x_filter) != len(keypoints):
            self.x_filter = list(
                map(
                    lambda _: OneEuroFilter(
                        self.frequency,
                        self.min_cutoff,
                        self.beta,
                        self.cutoff_threshold,
                        self.beta_threshold,
                        self.cutoff_derivative
                    ),
                    keypoints
                )
            )
            self.y_filter = list(
                map(
                    lambda _: OneEuroFilter(
                        self.frequency,
                        self.min_cutoff,
                        self.beta,
                        self.cutoff_threshold,
                        self.beta_threshold,
                        self.cutoff_derivative
                    ),
                    keypoints
                )
            )

    def reset(self):
        self.x_filter = None
        self.y_filter = None
