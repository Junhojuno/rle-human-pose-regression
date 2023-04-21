import os
import cv2
import argparse

if __package__ is None:
    import sys
    from os import path
    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.model import PoseRegModel
from src.inference.tracker import Tracker


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backbone', '-b',
        dest='backbone',
        default='resnet50',
        required=False
    )
    parser.add_argument(
        '--weights', '-w',
        dest='weights_path',
        required=True
    )
    parser.add_argument(
        '--video',
        dest='video_file',
        required=True
    )
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        required=True
    )
    parser.add_argument(
        '--smoothing',
        action='store_true'
    )
    return parser.parse_args()


def run_video_demo():
    """run video demo with frame limitation for benchmark purpose
    Args:
        frame_limit (int, optional): Defaults to 1000.
    """
    args = define_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    os.makedirs(args.save_dir, exist_ok=True)
    save_file = args.video_file.split('/')[-1].split('.')[0]
    save_file = \
        save_file + '_no_smoothing' if not args.smoothing else save_file
    save_file_path = os.path.join(
        args.save_dir, '{}.mp4'.format(save_file)
    )
    if os.path.exists(save_file_path):
        raise ValueError(f'{save_file_path} already exists.')

    cap = cv2.VideoCapture(args.video_file)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(save_file_path, fourcc, fps, (frame_w, frame_h))

    input_shape = [256, 192, 3]
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

    while True:
        success, frame = cap.read()
        if not success:
            print("no more images, close.")
            break
        image = tracker.run(frame, thr=0.1)
        writer.write(image)

    cap.release()
    writer.release()


if __name__ == "__main__":
    run_video_demo()
