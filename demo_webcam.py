import os
import cv2

from src.model import RLEModel
from src.inference.custom import Tracker


def run_webcam_demo():
    """run webcam demo with frame limitation for benchmark purpose
    Args:
        frame_limit (int, optional): Defaults to 1000.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    cam = cv2.VideoCapture(0)
    frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_shape = [256, 192, 3]
    num_keypoints = 17
    
    model = RLEModel(
        num_keypoints,
        input_shape,
        is_training=False
    )
    model.load_weights(
        'results/only_coco/basic_coco/rle/resnet50/b32x1_lr0.001_s2.0_sf0.25_r45/ckpt/best_model.tf'
    )
    tracker = Tracker(model, [frame_h, frame_w], input_shape[:2])

    while True:
        success, frame = cam.read()
        if not success:
            print("no more images, close.")
            break
        
        image = tracker.run(frame)
        cv2.imshow("RLE Realtime Demo", image)
        
        if cv2.waitKey(1) == ord("q"):
            break
        
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
     run_webcam_demo()
