import numpy as np
import cv2


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0.0, 0.0]),
    inv=0,
):
    scale_tmp = scale * 200
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (src_w - 1) * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def transform_preds(coords, center, scale, output_size):
    """transform prediction back using center and scale information
    Args:
        coords (np.array): predicted keypoint coordinate
        center (np.array): center position of the object
        scale (np.array): affine scaling factor with pixel std 200
        output_size (tuple): heatmap height and width
    Returns:
        target_coords (np.array): result coordinate
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for point in range(coords.shape[0]):
        target_coords[point, 0:2] = affine_transform(coords[point, 0:2], trans)
    return target_coords


def crop_image_to_portrait(image):
    """crop image to make portrait 4:3 shape
    Args:
        image (np.array): array like image
        Example: (480, 640, 3)
    Returns:
        image (np.array): cropped image
        Example: (480, 360, 3)
    """
    height, width = image.shape[:2]
    if height < width:  # wide frame
        quarter_height = height // 4
        result_gap = width - (quarter_height * 3)
        image = image[:, result_gap // 2 : width - result_gap // 2]
    elif height > width:  # long frame
        quarter_height = height // 4
        result_gap = height - (quarter_height * 3)
        image = image[result_gap // 2 : width - result_gap // 2, :]
    return image


def crop_image_to_portrait_v2(image):
    """crop image to make portrait 4:3 shape
    Args:
        image (np.array): array like image
        Example: (480, 640, 3)
    Returns:
        image (np.array): cropped image
        Example: (480, 360, 3)
    """
    height, width = image.shape[:2]
    if height < width:  # wide frame
        image_type = 'wide'
        expected_width = (height // 4) * 3
        gap = width - expected_width
        portrait_image = image[:, gap // 2 : width - gap // 2]  # make it shallow
    elif height > width:  # long frame
        expected_width = (height // 4) * 3
        gap = width - expected_width
        if gap == 0:
            image_type = 'long_perfect'
            portrait_image = image.copy()
        elif gap > 0:  # 4:3비율보다 4:4에 가까운 비율일때, height를 잘라낸다
            image_type = 'long_height'
            expected_width = (height // 4) * 3
            gap = width - expected_width
            portrait_image =  image[:, gap // 2 : width - gap // 2]
        else:  # gap < 0, 4:3비율보다 4:2에 가까운 비율일때, width를 잘라낸다.
            image_type = 'long_width'
            expected_height = (width // 3) * 4
            gap = height - expected_height
            portrait_image =  image[gap // 2 : height - gap // 2, :]  # make it shallow
    return portrait_image, image_type, gap


def crop_image_to_square(image):
    """crop image to make square shape
    Args:
        image (np.array): array like image
        Example: (480, 640, 3)
    Returns:
        image (np.array): cropped image
        Example: (480, 480, 3)
    """
    height, width = image.shape[:2]
    if height < width:
        half_diff = (width - height) // 2
        image = image[:, half_diff : width - half_diff]
    elif height > width:
        half_diff = (height - width) // 2
        image = image[half_diff : width - half_diff, :]
    return image


def crop_with_bbox(img, bbox, output_size):
    center, scale = xyxy_to_cs(bbox)
    trans = get_affine_transform(center, scale, 0, output_size)
    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])), flags=cv2.INTER_LINEAR
    )
    return dst_img, center, scale


def xyxy_to_xywh(bbox: list, img: dict = None) -> list:
    """convert xyxy bounding box to xywh format
    Args:
        bbox (list): ltrb(xyxy) formatted bounding box
        img (dict): optional image input to convert empty bbox
    Returns:
        bbox (list): converted bounding box
    """
    if None in bbox:
        return [0, 0, img["width"], img["height"]]

    x_min, y_min, x_max, y_max = bbox
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def xyxy_to_cs(bbox: list, aspect_ratio: float=0.75) -> tuple:
    """convert xyxy bounding box to center and scale
    Args:
        bbox (list): ltrb(xyxy) formatted bounding box
        aspect_ratio (float): image aspect ratio (width / height)
    Returns:
        center, scale (np.ndarray, float): center and scale values
    """
    x, y, w, h = xyxy_to_xywh(bbox)
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + (w - 1) * 0.5
    center[1] = y + (h - 1) * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200, h * 1.0 / 200], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


COCO_POINTS_LINKS = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [6, 8],
    [8, 10],
    [5, 7],
    [7, 9],
    [5, 11],
    [6, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [12, 11],
    [6, 5],
    [15, 19],
    [16, 21],
    [19, 17],
    [19, 18],
    [20, 22],
    [21, 22],
]
