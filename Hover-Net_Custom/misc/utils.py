import glob
import inspect
import logging
import os
import shutil

import cv2
import numpy as np
from scipy import ndimage

def cropping_center(x, crop_shape, batch=False):
    """
    입력 이미지 혹은 이미지 배치를 중앙에서부터 crop합니다.

    Args:
        x (np.ndarray): 입력 배열. 
            - 단일 이미지의 경우 (H, W, ...) 형태이며,
            - 배치의 경우 (N, H, W, ...) 형태로 기대합니다.
        crop_shape (tuple): (crop_height, crop_width) 형태의 crop 크기.
        batch (bool, optional): 입력 배열에 배치 차원이 있는지 여부. Defaults to False.

    Returns:
        np.ndarray: 중앙 crop된 이미지 또는 이미지 배치.
    """
    # 이미지 높이와 너비 추출
    if batch:
        height, width = x.shape[1:3]
    else:
        height, width = x.shape[:2]

    crop_h, crop_w = crop_shape

    # crop할 영역의 시작 좌표 계산 (중앙 정렬)
    start_h = (height - crop_h) // 2
    start_w = (width - crop_w) // 2

    if batch:
        return x[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
    else:
        return x[start_h:start_h + crop_h, start_w:start_w + crop_w]

def rm_n_mkdir(dir_path):
    """Remove directory if exists and create a new one."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]




####
def mkdir(dir_path):
    """Make directory."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


####
def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.

    Args:
        inst_map: input instance map
    
    Returns:
        array of centroids
    
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


####
def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output


####
def log_debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


####
def log_info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
