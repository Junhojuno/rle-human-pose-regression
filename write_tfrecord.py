from pycocotools.coco import COCO
import os
import numpy as np
import tensorflow as tf
from typing import List, Optional, Dict
from pathlib import Path


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte (list)."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double (list)."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint (list)."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(annot):
    feature = dict()
    image_raw = open(annot['image_file'], 'rb').read()
    feature['img_id'] = _int64_feature([annot['img_id']])
    feature['image_raw'] = _bytes_feature([image_raw])
    feature['bbox'] = _float_feature(annot['bbox'])
    feature['keypoints'] = _float_feature(annot['keypoints'])
    return tf.train.Example(features=tf.train.Features(feature=feature))


def exist_person_in_an_image(
    coco: COCO,
    img_id: int,
) -> bool:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    return len(anns) > 0


def check_n_keypoints_in_bbox(keypoints, bbox):
    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x2 = x1 + w
    y2 = y1 + h
    # vis > 0
    visible = keypoints[:, 2] > 0
    xy = keypoints[:, :2][visible]
    inners = (
        (xy[:, 0] >= x1) &
        (xy[:, 0] <= x2) &
        (xy[:, 1] >= y1) &
        (xy[:, 1] <= y2)
    ).astype('float')
    return np.sum(inners)


def get_filename(anno: Dict) -> str:
    filename = '{:012d}.jpg'.format(anno['image_id'])
    return filename


def reformat(
    anno: Dict,
    root: str,
    mode: str
) -> Dict:
    """서로 다른 데이터셋을 하나의 포맷으로 통일하기 위한 처리"""
    filename = get_filename(anno)
    new_anno = {
        'img_id': anno['image_id'],
        'image_file': os.path.join(root, 'images', f'{mode}2017', filename),
        'bbox': anno['bbox'],
        'keypoints': anno['keypoints'],
        'num_keypoints': anno['num_keypoints'],
    }
    return new_anno


def load_mscoco(coco_dir: Path, mode: str) -> List:
    coco_path = str(coco_dir / f'annotations/person_keypoints_{mode}2017.json')
    coco = COCO(coco_path)
    img_ids = list(
        filter(
            lambda img_id: exist_person_in_an_image(coco, img_id),
            coco.getImgIds()
        )
    )
    annots = []
    for img_id in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        for ann in anns:
            image_dic = coco.loadImgs([ann['image_id']])[0]
            image_height, image_width = image_dic['height'], image_dic['width']
            joints = ann['keypoints']
            if (ann['image_id'] not in coco.imgs) \
                    or ann['iscrowd'] \
                    or (np.sum(joints[2::3]) == 0) \
                    or (ann['num_keypoints'] == 0):
                continue

            # bbox 내에 keypoints가 하나도 없는 경우는 제외!
            n_kpts_in_bbox = check_n_keypoints_in_bbox(
                np.reshape(ann['keypoints'], [-1, 3]),
                ann['bbox']
            )
            if n_kpts_in_bbox > 0:
                x1, y1, w, h = ann['bbox']
                x1 = np.max((0, x1))
                y1 = np.max((0, y1))
                x2 = np.min((image_width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((image_height - 1, y1 + np.max((0, h - 1))))
                if ann['area'] > 0 and (x2 > x1 or y2 > y1):
                    ann['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                    annots.append(reformat(ann, coco_dir, mode))
    return annots


def convert_to_tfrecord(
    coco_dir: Path,
    mode: str,
    save_dir: str,
    shard_size: Optional[int] = 1024
):
    save_dir = os.path.join(save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    coco_annots = load_mscoco(coco_dir, mode)
    N = len(coco_annots)

    i = 0
    shard_count = 0
    while i < len(coco_annots):
        record_path = os.path.join(
            save_dir, f'{mode}_{N}_{shard_count:04d}.tfrecord'
        )
        with tf.io.TFRecordWriter(record_path) as writer:
            for j in range(shard_size):
                example = serialize_example(coco_annots[i])
                writer.write(example.SerializeToString())
                i += 1
                if i == len(coco_annots):
                    break
            if i >= len(coco_annots):
                break
        print('Finished writing', record_path)
        shard_count += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cwd', '-d',
        default='./',
        required=False,
        help='current working directory'
    )
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    coco_dir = cwd.parent / 'datasets' / 'mscoco'
    save_dir = coco_dir / 'tfrecords'

    convert_to_tfrecord(
        coco_dir,
        mode='train',
        save_dir=save_dir,
        shard_size=1024
    )
    # validation set
    convert_to_tfrecord(
        coco_dir,
        mode='val',
        save_dir=save_dir,
        shard_size=1024
    )
