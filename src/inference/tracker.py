"""new inference pipeline"""
from typing import List, Dict
import time
import cv2
import numpy as np

from src.inference.filter import (
    LowPassFilter,
    KeypointOneEuroFilter,
    SECOND_TO_MICRO_SECONDS
)

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}


class Tracker:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model,
        original_shape: List,
        target_shape: List = [128, 96],
        use_tracker: bool = False,
    ):
        self.model = model
        self.target_shape = target_shape
        self.target_h = target_shape[0]
        self.target_w = target_shape[1]
        self.target_ratio = self.target_w / self.target_h

        self.frame_w = original_shape[1]
        self.frame_h = original_shape[0]

        self.pad_info = None
        self.bbox = [0, 0, self.frame_w, self.frame_h]  # initial ROI

        self.timestamp = None

        if use_tracker:
            # ROI filter
            self.x1_filter = LowPassFilter(alpha=0.9)  # xmin
            self.y1_filter = LowPassFilter(alpha=0.9)  # ymin
            self.x2_filter = LowPassFilter(alpha=0.9)  # xmax
            self.y2_filter = LowPassFilter(alpha=0.9)  # ymax

            # keypoints filter
            self.keypoints_filter = KeypointOneEuroFilter(
                min_cutoff=1.5,
                beta=100.0
            )

        self.use_trakcer = use_tracker
        self.idx = 0

    def visualize(self, image, bbox, keypoints, alpha=0.5, thr=0.1):
        """draw keypoint on original image"""
        if not isinstance(image, np.ndarray):
            image = image.numpy()
        if isinstance(bbox, list):
            bbox = list(map(int, bbox))
        if not isinstance(keypoints, np.ndarray)\
                and isinstance(keypoints, list):
            keypoints = np.reshape(keypoints, [17, 3])

        for p, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if keypoints[p[0], 2] > thr and keypoints[p[1], 2] > thr:
                image = cv2.line(
                    image,
                    tuple(np.int32(np.round(keypoints[p[0], :2]))),
                    tuple(np.int32(np.round(keypoints[p[1], :2]))),
                    color=color,
                    thickness=2
                )
        image = cv2.addWeighted(image, alpha, image, 1 - alpha, 0)
        image = cv2.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            [0, 0, 255],
            thickness=2
        )
        return image

    def preprocess(self, frame):
        """crop -> pad -> resize"""
        cropped_image = self._crop_image_from_ROI(frame)
        padded_image = cv2.copyMakeBorder(
            cropped_image,
            self.pad_info['padT'],
            self.pad_info['padB'],
            self.pad_info['padL'],
            self.pad_info['padR'],
            cv2.BORDER_CONSTANT,
            (0, 0, 0)
        )
        self.scales = [
            padded_image.shape[1] / (self.target_w),
            padded_image.shape[0] / (self.target_h)
        ]
        input_image = cv2.resize(
            padded_image,
            (self.target_w, self.target_h)
        )
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype('float') / 255.
        input_image -= np.reshape(self.MEANS, [1, 1, 3])
        input_image /= np.reshape(self.STDS, [1, 1, 3])
        return input_image

    def predict(self, inputs) -> Dict:
        inputs = np.expand_dims(inputs, 0)
        mu, sigma = self.model(inputs, training=False)  # Dict

        mu = mu.numpy()[0]  # (K, 2)
        scores = np.mean(
            1 - sigma.numpy()[0],
            axis=-1,
            keepdims=True
        )  # (K, 1)

        outputs = np.concatenate([mu, scores], axis=-1)
        return outputs

    def postprocess(self, outputs):
        outputs[:, 0] = (outputs[:, 0] + 0.5) * self.target_w
        outputs[:, 1] = (outputs[:, 1] + 0.5) * self.target_h

        outputs[:, :2] *= [self.scales]
        outputs[:, :2] -= [[self.pad_info['padL'],
                            self.pad_info['padT']]]
        xmin, ymin, _, _ = self.bbox
        outputs[:, :2] += [[xmin, ymin]]

        if (self.timestamp is not None) and self.use_trakcer:
            outputs[:, :2] /= [[self.frame_w, self.frame_h]]
            outputs = self.keypoints_filter.apply(outputs, self.timestamp, 1)
            outputs[:, :2] *= [[self.frame_w, self.frame_h]]

        return outputs

    def run(self, frame, thr: float = 0.1):
        if self.use_trakcer:
            self.timestamp = time.time() * SECOND_TO_MICRO_SECONDS

        self.__determine_pad_info_from_bbox(self.bbox, self.target_ratio)

        inputs = self.preprocess(frame)
        outputs = self.predict(inputs)
        keypoints = self.postprocess(outputs)

        image = self.visualize(frame.copy(), self.bbox, keypoints, thr=thr)
        self.__update_ROI_from_prediction(keypoints, thr)

        if self.use_trakcer:
            self.__filter_ROI()
        return image

    def __update_ROI_from_prediction(self, prediction, thr) -> None:
        self.bbox = self._get_bbox_from_keypoints(
            prediction,
            thr,
            s=0.25,
            target_ratio=self.target_ratio
        )

    def __determine_pad_info_from_bbox(
        self,
        bbox: List,
        target_ratio: float = 0.75
    ) -> None:
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_ratio = bbox_width / bbox_height
        if bbox_ratio > target_ratio:  # 상대적으로 width가 넓다(height에 pad를 주자).
            # expected_height = (image_width / 3) * 4
            expected_height = bbox_width / target_ratio
            padT = padB = (expected_height - bbox_height) // 2
            padT = padT if padT > 0 else 0
            padB = padB if padB > 0 else 0
            padL, padR = 0, 0
        else:
            expected_width = bbox_height * target_ratio
            padL = padR = (expected_width - bbox_width) // 2
            padL = padL if padL > 0 else 0
            padR = padR if padR > 0 else 0
            padT, padB = 0, 0
        self.pad_info = {
            'padT': int(padT),
            'padB': int(padB),
            'padL': int(padL),
            'padR': int(padR),
        }

    def _crop_image_from_ROI(self, frame):
        x1, y1, x2, y2 = list(map(int, self.bbox))
        return frame[y1:y2, x1:x2, :]

    def __filter_ROI(self) -> None:
        if self.__is_initial_ROI():
            self.x1_filter.reset()
            self.y1_filter.reset()
            self.x2_filter.reset()
            self.y2_filter.reset()
        else:
            self.bbox[0] = self.x1_filter.apply(self.bbox[0])
            self.bbox[1] = self.y1_filter.apply(self.bbox[1])
            self.bbox[2] = self.x2_filter.apply(self.bbox[2])
            self.bbox[3] = self.y2_filter.apply(self.bbox[3])

    def __is_initial_ROI(self) -> bool:
        result =\
            (self.bbox[0] == 0) \
            and (self.bbox[1] == 0) \
            and (self.bbox[2] == self.frame_w) \
            and (self.bbox[3] == self.frame_h)
        return result

    def _get_bbox_from_keypoints(
        self,
        keypoints,
        thr: float = 0.1,
        s: float = 0.25,
        target_ratio: float = 0.75
    ) -> List:
        vis = keypoints[:, 2] > thr
        keypoints = keypoints[vis]
        if len(keypoints) < 2:
            return self.__get_initial_ROI()

        image_height, image_width = self.frame_h, self.frame_w

        xmin = np.amin(keypoints[:, 0])
        ymin = np.amin(keypoints[:, 1])
        xmax = np.amax(keypoints[:, 0])
        ymax = np.amax(keypoints[:, 1])

        xmin = xmin if xmin > 0 else 0
        ymin = ymin if ymin > 0 else 0
        if (xmax <= 0) or (xmax <= xmin) or (ymax <= 0) or (ymax <= ymin):
            return self.__get_initial_ROI()

        cx = (xmax + xmin) / 2
        cy = (ymax + ymin) / 2
        bbox_width = (xmax - xmin + 1)
        bbox_height = (ymax - ymin + 1)

        if (bbox_width * bbox_height) < 100:
            return self.__get_initial_ROI()

        bbox_height *= (1 + s)
        bbox_width *= (1 + s)

        if (bbox_width / bbox_height) > target_ratio:
            bbox_height = bbox_width / target_ratio
        else:
            bbox_width = bbox_height * target_ratio

        new_xmin = np.maximum(0, cx - bbox_width / 2)
        new_ymin = np.maximum(0, cy - bbox_height / 2)
        new_xmax = np.minimum(image_width, cx + bbox_width / 2)
        new_ymax = np.minimum(image_height, cy + bbox_height / 2)
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    def __get_initial_ROI(self) -> List:
        return [0., 0., float(self.frame_w), float(self.frame_h)]
