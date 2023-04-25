"""export to tflite"""
from pathlib import Path
from typing import Dict
import argparse
import tempfile

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras import layers

from src.model import PoseRegModel


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
        '--subtitle',
        default='basic',
        required=False,
        help='학습한 모델의 이름이 같은데 다른 특징을 가지고 있는 경우,'
        '구분하기 위해 subtitle을 추가'
    )
    return parser.parse_args()


def create_deploy_model(args):
    input_shape = [args.input_height, args.input_width, 3]
    inputs = Input(input_shape, dtype=tf.uint8)
    preprocess_layer = Sequential(
        [
            layers.Rescaling(scale=1./255.),
            layers.Normalization(
                -1,
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ], name='normalize_input'
    )
    model = PoseRegModel(
        args.num_keypoints,
        input_shape,
        args.backbone
    )
    model.load_weights(args.weights)
    outputs = preprocess_layer(inputs)
    outputs = model(outputs)
    return Model(inputs, outputs, name=f'rle_{args.backbone}_deploy')


def convert_to_tflite(
    args: Dict,
    saved_model_dir: str,
    save_tflite_dir: Path
) -> None:
    """convert tf model to tflite as float16"""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    # converter.allow_custom_ops = True
    tflite_quant_model = converter.convert()

    filename = '{height}x{width}_rle_{backbone}_ptq_f16.tflite'.format(
        height=args.input_height,
        width=args.input_width,
        backbone=args.backbone
    )
    save_file = save_tflite_dir / filename

    save_file.write_bytes(tflite_quant_model)


def main():
    args = define_argparser()
    cwd = Path('.').resolve()
    save_tflite_dir = \
        cwd \
        / 'tflites' \
        / args.subtitle \
        / f'{args.input_height}x{args.input_width}_{args.backbone}'

    save_tflite_dir.mkdir(parents=True, exist_ok=True)

    model = create_deploy_model(args)

    # save tf-model and convert to tflite
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        convert_to_tflite(args, tmpdir, save_tflite_dir)


if __name__ == '__main__':
    main()
