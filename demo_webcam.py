import os
import cv2

from src.model import RLEModel
from src.tracker.projector import PoseProjector


def run_webcam_demo(sec: int = None):
    """run webcam demo with frame limitation for benchmark purpose
    Args:
        frame_limit (int, optional): Defaults to 1000.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    frame_limit = None
    if sec is not None:
        frame_limit = sec * 30

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
    cam = cv2.VideoCapture(0)
    projector = PoseProjector(cam, model, input_shape)
    projector.start_timer()

    while projector.cam.isOpened():
        image = projector.read()
        if image is None:
            print("no more images, close.")
            break
        image = projector.draw(image)
        cv2.imshow("Webcam Demo", image)

        if cv2.waitKey(1) == ord("q"):
            projector.stop_timer()
            break
        
        if frame_limit:
            turn_off = (projector.frames_processed == frame_limit)
            if turn_off:
                break

    projector.cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()