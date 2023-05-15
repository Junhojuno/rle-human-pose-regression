import os
import cv2
import argparse
import time

if __package__ is None:
    import sys
    from os import path

    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.model import PoseRegModel
from src.inference.tracker import Tracker
from src.inference.fps import calculate_fps, draw_fps


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", "-b",
        dest="backbone",
        default="resnet50",
        required=False
    )
    parser.add_argument(
        "-height",
        dest='height',
        type=int,
        default=256,
        required=False,
        help="input height"
    )
    parser.add_argument(
        "-width",
        dest='width',
        type=int,
        default=192,
        required=False,
        help="input width"
    )
    parser.add_argument(
        "--weights", "-w",
        dest="weights_path",
        required=True
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="use postprocess filters or not"
    )
    return parser.parse_args()


def run_webcam_demo(fps_avg_frame_count: int = 10):
    """run webcam demo with frame limitation for benchmark purpose
    Args:
        frame_limit (int, optional): Defaults to 1000.
    """
    args = define_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    cam = cv2.VideoCapture(0)
    frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_shape = [args.height, args.width, 3]
    num_keypoints = 17

    model = PoseRegModel(
        num_keypoints,
        input_shape,
        backbone_type=args.backbone
    )
    model.load_weights(args.weights_path)
    tracker = Tracker(
        model,
        [frame_h, frame_w],
        input_shape[:2],
        args.smoothing
    )

    counter, fps = 0, 0
    start_time = time.time()
    while True:
        success, frame = cam.read()
        if not success:
            print("no more images, close.")
            break
        counter += 1

        image = tracker.run(frame, thr=0.1)
        if counter % fps_avg_frame_count == 0:
            fps = calculate_fps(start_time, fps_avg_frame_count)
            start_time = time.time()

        # Show the FPS
        image = draw_fps(image, fps)

        cv2.imshow("RLE Realtime Demo", image)

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
