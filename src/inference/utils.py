"""Supporting Modules"""
import numpy as np


KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}


def extract_next_roi(bbox, target_ratio=0.75):
    """bbox = [x1, y1, x2, y2]"""
    roi = bbox
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_ratio = bbox_width / bbox_height
    if bbox_ratio > target_ratio:  # 상대적으로 width가 넓다(height에 pad를 주자).
        # expected_height = (image_width / 3) * 4
        expected_height = bbox_width / target_ratio
        padT = padB = (expected_height - bbox_height) // 2
        padL, padR = 0, 0
    else:
        expected_width = bbox_height * target_ratio
        padL = padR = (expected_width - bbox_width) // 2
        padT, padB = 0, 0
    return {
        'roi': roi,
        'padT': padT,
        'padB': padB,
        'padL': padL,
        'padR': padR,
    }


def extract_init_roi(image, target_ratio=0.75):
    """정보가 없는 경우에는 전체 이미지를 target 비율로 맞춰준다!"""
    image_height, image_width = image.shape[:2]
    ratio = image_width / image_height
    
    roi = [0, 0, image_width, image_height]
    if ratio > target_ratio:  # 상대적으로 width가 넓다(height에 pad를 주자).
        # expected_height = (image_width / 3) * 4
        expected_height = image_width / target_ratio
        padT = padB = (expected_height - image_height) // 2
        padL, padR = 0, 0
    else:
        expected_width = image_height * target_ratio
        padL = padR = (expected_width - image_width) // 2
        padT, padB = 0, 0
    return {
        'roi': roi,
        'padT': padT,
        'padB': padB,
        'padL': padL,
        'padR': padR,
    }


def zero_pad_to_image(image, padT, padB, padL, padR):
    padT, padB, padL, padR = list(map(int, [padT, padB, padL, padR]))
    if padT:
        padding = np.zeros_like(image[:padT, :, :])
        image = np.vstack([padding, image])
    if padB:
        padding = np.zeros_like(image[:padB, :, :])
        image = np.vstack([image, padding])
    if padL:
        padding = np.zeros_like(image[:, :padL, :])
        image = np.hstack([padding, image])
    if padR:
        padding = np.zeros_like(image[:, :padR, :])
        image = np.hstack([image, padding])
    return image


def crop_and_pad(image, crop_region):
    xmin, ymin, xmax, ymax = list(map(int, crop_region['roi']))
    # image = image[ymin:ymax, xmin:xmax, :]
    image = image[ymin:, xmin:xmax, :]
    image = zero_pad_to_image(
        image,
        crop_region['padT'],
        crop_region['padB'],
        crop_region['padL'],
        crop_region['padR']
    )
    return image


def strip_pad(image, padT, padB, padL, padR):
    padT, padB, padL, padR = list(map(int, [padT, padB, padL, padR]))
    if padT:
        image = image[padT:, :, :]
    if padB:
        image = image[:-padB, :, :]
    if padL:
        image = image[:, padL:, :]
    if padR:
        image = image[:, :-padR, :]
    return image


def get_bbox_from_keypoints(keypoints, image_shape, thr=0.1, s=0.2):
    vis = keypoints[:, 2] > thr
    keypoints = keypoints[vis]
    if len(keypoints) < 2:
        return None
    
    image_height, image_width = image_shape
    
    xmin = np.amin(keypoints[:, 0])
    ymin = np.amin(keypoints[:, 1])
    xmax = np.amax(keypoints[:, 0])
    ymax = np.amax(keypoints[:, 1])
    
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    # bbox_width = (xmax - xmin + 1) * (1 + s)
    # bbox_height = (ymax - ymin + 1) * (1 + s)
    bbox_width = (xmax - xmin + 1)
    bbox_height = (ymax - ymin + 1)
    
    if bbox_width > bbox_height:
        h_s = s
        w_s = s / 2
    elif bbox_width < bbox_height:
        h_s = s / 2
        w_s = s
    else:
        h_s = s
        w_s = s
    
    new_xmin = np.maximum(0, cx - bbox_width / 2 * (1 + w_s))
    new_ymin = np.maximum(0, cy - bbox_height / 2 * (1 + h_s))
    new_xmax = np.minimum(image_width, cx + bbox_width / 2 * (1 + w_s))
    new_ymax = np.minimum(image_height, cy + bbox_height / 2 * (1 + h_s))
    return [new_xmin, new_ymin, new_xmax, new_ymax]
