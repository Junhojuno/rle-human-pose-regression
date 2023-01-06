import os
from pathlib import Path
import cv2
import argparse
from typing import List

from src.model import RLEModel
from src.tracker.projector import PoseProjector


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_height',
                        dest='input_height',
                        required=False,
                        default=256)
    parser.add_argument('--input_width',
                        dest='input_width',
                        required=False,
                        default=192)
    parser.add_argument('--num_keypoints',
                        dest='num_keypoints',
                        required=False,
                        default=17)
    parser.add_argument('--thr',
                        dest='thr',
                        required=False,
                        default=0.3)
    parser.add_argument('--weights',
                        dest='weights',
                        required=False,
                        default='results/only_coco/basic_coco/rle/resnet50/b32x1_lr0.001_s2.0_sf0.25_r45/ckpt/best_model.tf')
    parser.add_argument('--video',
                        dest='video',
                        required=True,
                        help='relative file path')
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        required=False,
                        default='results_demo/demo_videos')
    parser.add_argument('--sec',
                        dest='sec',
                        type=int,
                        required=False,
                        default=None)
    return parser.parse_args()



def get_video_writer(
    cam, frame_width, frame_height, save_path='webcam_demo.mp4', sec=None
):
    w = frame_width
    h = frame_height
    fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    if sec is None:  # fully
        target_frames = None
    else:
        target_frames = int(fps * sec)
    return writer, target_frames


def run_video_demo(args):
    """run webcam demo with frame limitation for benchmark purpose
    Args:
        frame_limit (int, optional): Defaults to 1000.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    input_shape = [args.input_height, args.input_width, 3]
    num_keypoints = args.num_keypoints
    
    model = RLEModel(
        num_keypoints,
        input_shape,
        is_training=False
    )
    model.load_weights(args.weights)
    cam = cv2.VideoCapture(args.video)
    
    filename = args.video.split('/')[-1]
    save_file_path = os.path.join(args.save_dir, filename)
        
    projector = PoseProjector(cam, model, input_shape)
    writer, frame_limit = get_video_writer(cam, projector.img_w, projector.img_h, save_file_path, args.sec)

    projector.start_timer()

    while projector.cam.isOpened():
        image = projector.read()
        if image is None:
            print("no more images, close.")
            break
        image = projector.draw(image)
        writer.write(image)
 
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
    args = define_argparser()
    run_video_demo(args)