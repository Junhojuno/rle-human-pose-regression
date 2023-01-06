import tensorflow as tf
import cv2
import time
import numpy as np
from typing import List

from src.tracker.misc import (
    transform_preds,
    crop_image_to_square,
    crop_with_bbox,
    COCO_POINTS_LINKS,
    crop_image_to_portrait,
    crop_image_to_portrait_v2
)


class Projector:
    """Parent interface for visualization and inference"""

    def __init__(self, cam: cv2.VideoCapture, crop_mode: str = "portrait"):
        # initialize parameters
        self.cam = cam
        
        if crop_mode == "portrait":
            self.read = self.capture_portrait
        elif crop_mode == "square":
            self.read = self.capture_squared
        
        self.init_image = self.read()
        self.img_h = self.init_image.shape[0]
        self.img_w = self.init_image.shape[1]
        print("camera initialized with size:", self.init_image.shape)

    def capture_portrait(self, mirror_mode: bool = True):
        """capture image cropped in portrait
        
        Args:
            image (np.array): cropped image
            mirror_mode (bool, optional): 
                when you raise your right hand, the display will raise right hand
        """
        success, frame = self.cam.read()
        if not success:
            return None
        self.time_read = time.time()
        portrait, self.image_type, self.gap = crop_image_to_portrait_v2(frame)
        # portrait = crop_image_to_portrait(frame)
        if mirror_mode:
            portrait = cv2.flip(portrait, 1)
        return portrait

    def capture_squared(self, mirror_mode: bool = True):
        """capture image cropped in square
        
        Args:
            image (np.array): cropped image
            mirror_mode (bool, optional): 
                when you raise your right hand, the display will raise right hand
        """
        _, image = self.cam.read()
        image = crop_image_to_square(image)
        if mirror_mode:
            image = cv2.flip(image, 1)
        return image

    def start_timer(self):
        """initialize current execution time """
        self.frames_processed = 0
        self.tick = time.time()

    def stop_timer(self):
        """summarize execution time """
        time_elapsed = time.time() - self.tick
        print(
            "Estimated FPS: %s (Total %sframes / %ssecs)"
            % ((self.frames_processed / time_elapsed), self.frames_processed, time_elapsed,)
        )


class PoseProjector(Projector):
    """Pose projector for human keypoint drawing and online demo"""

    def __init__(self, cam: cv2.VideoCapture, model: tf.keras.Model, input_shape: List, crop_mode="portrait"):
        super().__init__(cam=cam, crop_mode=crop_mode)
        self.model = model
        self.input_h = input_shape[0]
        self.input_w = input_shape[1]
        self.bbox = [0, 0, self.img_w, self.img_h]

    @tf.function
    def inference(self, model_input):
        return self.model(model_input, training=False)

    def preprocess(
        self,
        image: np.ndarray,
        bbox: List,
        means: List = [0.485, 0.456, 0.406],
        stds: List = [0.229, 0.224, 0.225]
    ) -> np.ndarray:
        image, self.center, self.scale = crop_with_bbox(
            image,
            bbox,
            output_size=(self.input_w, self.input_h)
        )
        # convert bgr channel oredering to rgb
        model_input = image[:, :, ::-1].astype('float')
        model_input /= 255.
        model_input -= [[means]]
        model_input /= [[stds]]

        # expand batch channel
        model_input = np.expand_dims(model_input, axis=0)
        return model_input

    def postprocess(self, outputs):
        """Up-scale prediction to 4:3 cropped image"""
        # transpose Batch x Height x Width x Channel (BHWC) to
        # Batch x Channel x Height x Width
        pred_kpts = outputs.mu.numpy()[0]
        pred_scores = outputs.maxvals.numpy()[0]
        # scale to image size
        pred_kpts[:, 0] = (pred_kpts[:, 0] + 0.5) * self.input_w
        pred_kpts[:, 1] = (pred_kpts[:, 1] + 0.5) * self.input_h
        
        pred_kpts = transform_preds(
            pred_kpts, self.center, self.scale, [self.input_w, self.input_h]
        )
        return pred_kpts.tolist(), pred_scores.tolist()

    def get_annotations(self, image, bbox):
        """simple pipeline from preprocessing to postprocessing"""
        model_input = self.preprocess(image, bbox)
        output = self.inference(model_input)
        keypoints, probs = self.postprocess(output)
        return keypoints, probs

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: List,
        probs: List,
        thr: float = 0.1,
        draw: bool = True
    ):
        """draw keypoint skeletons above threshold"""
        drawing_points = []
        if len(probs) == 17:
            links = COCO_POINTS_LINKS[:len(probs)-1]
        else:
            links = COCO_POINTS_LINKS

        for prob, keypoint in zip(probs, keypoints):
            if prob[0] > thr:
                if draw:
                    image = cv2.circle(
                        image,
                        (int(keypoint[0]), int(keypoint[1])),
                        3,
                        (0, 255, 255),
                        thickness=-1,
                        lineType=cv2.FILLED,
                    )

                # Add the point to the list if the probability is greater than the thr
                drawing_points.append((int(keypoint[0]), int(keypoint[1])))
            else:
                drawing_points.append(None)

        # Draw Skeleton
        for pair in links:
            if drawing_points[pair[0]] and drawing_points[pair[1]]:
                image = cv2.line(
                    image,
                    drawing_points[pair[0]],
                    drawing_points[pair[1]],
                    (0, 200, 100),
                    thickness=2,
                )
        self.frames_processed += 1
        return image

    def display_fps(self, image: np.ndarray):
        fps = int(1 / (time.time() - self.time_read))
        fps = f'FPS: {fps}'
        # display
        return cv2.putText(image, fps, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    def draw(self, image: np.ndarray, thr: float = 0.1):
        """draw human keypoints with paired points"""
        keypoints, probs = self.get_annotations(image, self.bbox)
        image = self.draw_skeleton(image.copy(), keypoints, probs, thr=thr)
        image = self.display_fps(image.copy())
        return image
