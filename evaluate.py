from pathlib import Path
import argparse

from src.model import PoseRegModel
from src.function import validate
from src.eval import load_eval_dataset


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_height', '-ih',
        dest='input_height',
        default=256,
        type=int,
        required=False
    )
    parser.add_argument(
        '--input_width', '-iw',
        dest='input_width',
        default=192,
        type=int,
        required=False
    )
    parser.add_argument(
        '--num_keypoints', '-k',
        dest='num_keypoints',
        default=17,
        type=int,
        required=False
    )
    parser.add_argument(
        '--backbone', '-b',
        dest='backbone',
        default='resnet50',
        type=str,
        required=False
    )
    parser.add_argument(
        '--weights', '-w',
        dest='weights',
        required=True
    )
    parser.add_argument(
        '--flip',
        dest='use_flip',
        action='store_true'
    )
    return parser.parse_args()


def main():
    args = define_argparser()
    cwd = Path('.').resolve()

    input_shape = [args.input_height, args.input_width, 3]
    num_keypoints = args.num_keypoints
    backbone = args.backbone
    weight_path = args.weights

    model = PoseRegModel(
        num_keypoints,
        input_shape,
        backbone
    )
    model.load_weights(weight_path)

    eval_ds = load_eval_dataset(
        str(
            cwd.parent
            / 'datasets'
            / 'mscoco'
            / 'tfrecords/val/*.tfrecord'
        ),
        32,
        num_keypoints,
        input_shape
    )
    _, _ = validate(
        model,
        eval_ds,
        input_shape,
        str(
            cwd.parent
            / 'datasets'
            / 'mscoco'
            / 'annotations'
            / 'person_keypoints_val2017.json'
        ),
        f'{args.input_height}x{args.input_width}_{backbone}',
        print,
        use_flip=args.use_flip
    )


if __name__ == '__main__':
    main()
