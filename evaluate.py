from pathlib import Path

from src.model import RLEModel
from src.eval.coco import load_coco_eval_dataset, eval_coco
from src.eval.utils import print_name_value


def main():
    cwd = Path('.').resolve()

    input_shape = [256, 192, 3]
    num_keypoints = 17
    weight_path = 'results/only_coco/basic_coco/rle/resnet50/b32x1_lr0.001_s2.0_sf0.25_r45/ckpt/best_model.tf'

    model = RLEModel(
        num_keypoints,
        input_shape,
        'resnet50',
        is_training=True
    )
    model.load_weights(weight_path)

    eval_ds = load_coco_eval_dataset(
        str(cwd.parent / 'datasets'),
        input_shape,
        num_keypoints,
        batch_size=32
    )
    ap, details = eval_coco(
        eval_ds,
        model,
        input_shape,
        use_flip=True,
    )
    print_name_value(details, 'rle_model', print_fn=print)


if __name__ == '__main__':
    main()
