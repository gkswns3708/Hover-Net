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
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)