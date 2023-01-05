"""new version of demo_images"""
import os
import cv2
import argparse
from tqdm import tqdm
from typing import List
import tensorflow as tf
import cv2
import numpy as np

from src.transforms import parse_example, preprocess
from src.model import RLEModel
from src.utils import visualize_with_RGB as visualize


def define_argparser():
    parser = argparse.ArgumentParser('RLEModel demo')
    parser.add_argument('--weights', '-w', dest='weight_path', required=True)
    # parser.add_argument('--input_', '-s', dest='input_size', default=192, help='model input resolution')
    parser.add_argument('--num_k', '-k', dest='num_keypoints', default=17, help='number of target keypoints')
    parser.add_argument('--vis_threshold', '-thr', dest='vis_threshold', type=float, default=0.3, help='keypoints visualization threshold')
    return parser.parse_args()


def load_pose_dataset_demo(
    file_pattern: str,
    num_keypoints: int,
    input_shape: List,
    use_norm: bool = True,
    use_aug: bool = True
):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)
    
    ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda record: parse_example(record, num_keypoints=num_keypoints),
        num_parallel_calls=AUTOTUNE
    )
    
    ds = ds.map(
        lambda image, bbox, keypoints: preprocess(image, bbox, keypoints,
                                                  use_image_norm=use_norm,
                                                  input_shape=input_shape,
                                                  use_aug=use_aug),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(1, drop_remainder=True, 
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return ds


def denormalize_image(image, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    image *= [[stds]]
    image += [[means]]
    image *= 255.
    return image


def main():
    args = define_argparser()
    
    save_folder = os.path.join('./samples', 'demo_images')
    os.makedirs(save_folder, exist_ok=True)
    
    input_shape = [256, 192, 3]
    file_pattern = '/home/alyce/alycehealth/datasets/wandb_test/val/tfrecords/*.tfrecords'
    ds = load_pose_dataset_demo(
        file_pattern,
        args.num_keypoints,
        input_shape,
        use_norm=True,
        use_aug=False
    )
    model = RLEModel(
        17,
        [256, 192, 3],
        is_training=False
    )
    model.load_weights(args.weight_path)
    
    for idx, (imgs, _) in tqdm(enumerate(ds.take(100)), '[demo images]'):
        outputs = model(imgs, training=False)
        pred_kpts = outputs.mu.numpy()
        pred_scores = outputs.maxvals.numpy()
        # scale to image size
        pred_kpts[:, :, 0] = (pred_kpts[:, :, 0] + 0.5) * input_shape[1]
        pred_kpts[:, :, 1] = (pred_kpts[:, :, 1] + 0.5) * input_shape[0]
        outputs = np.concatenate([pred_kpts, pred_scores], axis=-1)[0]
        
        imgs = denormalize_image(imgs)
        
        vis_image = visualize(cv2.cvtColor(imgs[0].numpy(), cv2.COLOR_RGB2BGR), outputs)
        
        cv2.imwrite(
            os.path.join(save_folder, f'{idx:012d}.jpg'),
            vis_image
        )


if __name__ == '__main__':
    main()
