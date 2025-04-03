import math
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils import cropping_center


class PatchExtractor(object):
    """
     Extract to generate pathces w/w.o padding
     Turn on debug mode to see how it is done.
    """
    # MO def __init__(self, win_size, step_size, debug=False): 
    # MO patch_type is added to the __init__ method
    def __init__(self, win_size, step_size, patch_type="mirror", debug=False):
        # 
        self.patch_type = patch_type
        self.win_size = win_size
        self.step_size = step_size
        self.debug = debug
        self.counter = 0  # Quick Memo - 2021-07-29 14:00:00
    
    def __get_patch(self, x, ptx):
        """
        Extract a patch from the image
        
        Args:
            x (np.ndarray): input image
            ptx (tuple): starting point of the patch(X, Y)
        """
        pty = (ptx[0] + self.win_size[0], ptx[1] + self.win_size[1])
        win = x[ptx[0] : pty[0], ptx[1] : pty[1]]
        assert (
            win.shape[0] == self.win_size[0] and win.shape[1] == self.win_size[1]
        ), "[BUG] Incorrect Patch Size {0}".format(win.shape)
        if self.debug:
            if self.patch_type == "mirror":
                cen = cropping_center(win, self.step_size)
                cen = cen[..., self.counter % 3]
                cen.fill(150)
            cv2.rectangle(x, ptx, pty, (255, 0, 0), 2)
            plt.imshow(x)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            self.counter += 1
        return win

    def __extract_valid(self, x):
        im_h = x.shape[0]
        im_w = x.shape[1]
        
        # Check if the last patch is smaller than the window size and calculate the last step point(X, Y)
        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            return flag, last_step
        
        # *_flag: if the last patch is smaller than the window size
        # *_last: the last patch's starting point
        h_flag, h_last = extract_infos(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.win_size[1], self.step_size[1])
        
        if h_flag:
            h_last = im_h - self.win_size[0]
        
        sub_patches = []
        
        # extract normal patches 
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        
        # extract the last row of patches
        if h_flag:
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (im_h - self.win_size[0], col))
                sub_patches.append(win)
        # extract the last column of patches
        if w_flag:
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, im_w - self.win_size[1]))
                sub_patches.append(win)
        # extract the last patch
        if h_flag and w_flag:
            win = self.__get_patch(x, (im_h - self.win_size[0], im_w - self.win_size[1]))
            sub_patches.append(win)
        return sub_patches
        
    
    def __extract_mirror(self, x):
        """
            Extract patches with mirror padding to handle the boundary problem.
            Args:
                x (np.ndarray): input image
            Return:
                a list of sub patches, each patch is same dtype as x
        """
        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt # Bottom padding size
        
        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl # Right padding size
        
        pad_type = "constant" if self.debug else "reflect"
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        
        # debug 
        if self.debug:
            label_list = []
            for patches in sub_patches:
                label_list.extend(list(np.unique(patches[:, :, 4])))
            print(f"최종 patch의 label 갯수: {set(label_list)}")
            print(f"Extracted {len(sub_patches)} patches")
        
        return sub_patches
    
    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        if patch_type == "mirror":
            return self.__extract_mirror(x)

        return self._extract(x)