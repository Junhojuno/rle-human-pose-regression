"""new inference pipeline"""
import cv2
import numpy as np

from src.inference.utils import (
    extract_init_roi,
    extract_next_roi,
    zero_pad_to_image,
    crop_and_pad,
    get_bbox_from_keypoints,
    KEYPOINT_EDGE_INDS_TO_COLOR
)


class Tracker:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    
    def __init__(self, model, original_shape, target_shape=[288, 224]):
        self.model = model
        self.target_shape = target_shape
        self.target_h = target_shape[0]
        self.target_w = target_shape[1]
        self.target_ratio = self.target_w / self.target_h
        self.crop_region = None
        
        self.frame_w = original_shape[1]
        self.frame_h = original_shape[0]

        self.cnt = 0

    def preprocess(self, frame):
        """
        frame -> ROI cropped image -> input image
        """
        if self.crop_region is None:
            self.crop_region = extract_init_roi(frame, self.target_ratio)
            padded_frame = zero_pad_to_image(
                frame.copy(),
                self.crop_region['padT'],
                self.crop_region['padB'],
                self.crop_region['padL'],
                self.crop_region['padR']
            )
            self.scales = [
                padded_frame.shape[1],
                padded_frame.shape[0],
            ]
            input_image = cv2.resize(
                padded_frame,
                [self.target_w, self.target_h]
            )
        else:
            cropped_padded_image = crop_and_pad(frame.copy(), self.crop_region)
            self.scales = [
                cropped_padded_image.shape[1],
                cropped_padded_image.shape[0],
            ]
            input_image = cv2.resize(
                cropped_padded_image,
                [self.target_w, self.target_h]
            )
        
        input_image = input_image[:, :, ::-1].astype('float')
        input_image /= 255.
        input_image -= [[self.MEANS]]
        input_image /= [[self.STDS]]
        return np.expand_dims(input_image, 0)
    
    def predict(self, inputs):
        outputs = self.model(inputs, training=False)
        return outputs
    
    def postprocess(self, outputs: np.ndarray):
        """upscaling outputs
        
        (heatmap) -> keypoints(0~heatmap_size) -> keypoints(0~cropped_size)
        """
        # keypoints[:, :2] *= [self.scales] # upscaling to size before resize
        keypoints = np.concatenate(
            [outputs.mu.numpy()[0], outputs.maxvals.numpy()[0]],
            axis=-1
        )
        # keypoints[:, 0] = (keypoints[:, 0] + 0.5) * self.scales[0]
        # keypoints[:, 1] = (keypoints[:, 1] + 0.5) * self.scales[1]
        keypoints[:, 0] = (keypoints[:, 0] + 0.5) * self.target_w
        keypoints[:, 1] = (keypoints[:, 1] + 0.5) * self.target_h
        if self.crop_region is None:
            keypoints[:, :2] -= [
                [self.crop_region['padL'], self.crop_region['padT']]
            ]
            # update crop_region
            bbox = get_bbox_from_keypoints(
                keypoints,
                [self.frame_h, self.frame_w],
                thr=0.1,
                s=0.8
            )
        else:
            keypoints[:, :2] -= [
                [self.crop_region['padL'], self.crop_region['padT']]
            ]
            xmin, ymin, _, _ = self.crop_region['roi']
            keypoints[:, :2] += [[xmin, ymin]]
            bbox = get_bbox_from_keypoints(
                keypoints,
                [self.frame_h, self.frame_w],
                thr=0.1,
                s=0.8
            )
        return keypoints, bbox
    
    def visualize(self, image, bbox, keypoints, alpha=0.5, thr=0.1):
        """draw keypoint on original image"""
        if not isinstance(image, np.ndarray):
            image = image.numpy()
        if isinstance(bbox, list):
            bbox = list(map(int, bbox))
        if not isinstance(keypoints, np.ndarray):
            keypoints = keypoints.numpy()

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
            image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0, 0, 255], 2
        )
        return image
    
    def reset(self):
        self.crop_region = None
    
    def run(self, frame):
        inputs = self.preprocess(frame)
        outputs = self.predict(inputs)
        keypoints, bbox = self.postprocess(outputs)
        if bbox is not None:
            self.crop_region = extract_next_roi(bbox, self.target_ratio)
        else:
            bbox = [0, 0, self.frame_w, self.frame_h]
            self.crop_region = extract_init_roi(frame, self.target_ratio)
        image = self.visualize(frame.copy(), bbox, keypoints)
        return image
