"""
단일 영상을 입력으로 받아 추론 결과를 해당 영상에 입혀서 새로운 영상을 생성
**get_video_writer의 argument 'sec'를 조절하여 출력 결과 영상의 길이 조절
"""
import os
import cv2
import argparse

from core.tracker.single_pose import SinglePoseTracker
from core.tracker.benchmark import BenchmarkMoveNetTracker
# from core.tracker.utils import visualize
from core.utils.visualize import visualize_with_RGB


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        dest='backbone',
                        required=False,
                        default='mobilenetv2')
    parser.add_argument('--input_size',
                        dest='input_size',
                        required=False,
                        default=192)
    parser.add_argument('--num_keypoints',
                        dest='num_keypoints',
                        required=False,
                        default=17)
    parser.add_argument('--threshold',
                        dest='vis_threshold',
                        required=False,
                        default=0.3)
    parser.add_argument('--weights',
                        dest='weights',
                        required=False,
                        default=None,
                        help='benchmark 모델을 사용하는게 아니라면, 필수!')
    parser.add_argument('--video',
                        dest='video_file',
                        required=True,
                        help='relative file path')
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        required=True)
    parser.add_argument('--sec',
                        dest='sec',
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument('--smoothing',
                        dest='use_smoothing',
                        action='store_true')
    parser.add_argument('--benchmark',
                        dest='use_benchmark',
                        action='store_true')
    return parser.parse_args()


def get_video_writer(cam, save_path='webcam_demo.mp4', sec=None):
    w = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    if sec is None:  # fully
        target_frames = None
    else:
        target_frames = int(fps * sec)
    return writer, target_frames


def display_fps(vis_image, fps=None):
    fps = f'FPS: {fps}'
    # display
    return cv2.putText(vis_image, fps, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


def main(args):
    if args.use_benchmark:
        # tracker = BenchmarkMoveNetTracker('movenet_lightning')
        tracker = BenchmarkMoveNetTracker('movenet_thunder')
    else:
        assert args.weights is not None, 'pretrained weight is needed!'
        tracker = SinglePoseTracker(
            weight_path=args.weights,
            backbone_type=args.backbone,
            input_shape=[args.input_size, args.input_size, 3],
            num_keypoints=args.num_keypoints,
            min_cutoff=0.5,
            beta=10.0
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cap = cv2.VideoCapture(args.video_file)

    save_file = args.video_file.split('/')[-1].split('.')[0]
    save_file = save_file + '_benchmark' if args.use_benchmark else save_file
    save_file_path = os.path.join(args.save_dir, '{}.mp4'.format(save_file))
    if os.path.exists(save_file_path):
        return

    writer, target_frames = get_video_writer(cap, save_file_path, args.sec)

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("no more images, close.")
            break
        # run inference
        detected_person = tracker.run(frame, use_smoothing=args.use_smoothing)

        # frame = visualize(frame, detected_person, keypoint_threshold=args.vis_threshold)
        frame = visualize_with_RGB(frame, detected_person)
        writer.write(frame)

        if target_frames is not None and frame_count > target_frames:
            break

        frame_count += 1

    cap.release()
    writer.release()


if __name__ == '__main__':
    args = define_argparser()
    main(args)
