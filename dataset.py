import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """
        Abstract class for interface of subsequent classes.
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError

class __CoNSeP(__AbstractDataset):
    """
        Defines the CoNSeP dataset as originally introduced in:
        Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical image analysis 58 (2019): 101563.
    """
    
    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    def load_ann(self, path, with_type=False):
        ann_inst = sio.loadmat(path)["inst_map"]
        full_ann_type = sio.loadmat(path)["type_map"]
        merged_ann_type = full_ann_type.copy()
        merged_ann_type[(full_ann_type == 3) | (full_ann_type == 4)] = 3
        merged_ann_type[(full_ann_type == 5) | (full_ann_type == 6) | (full_ann_type == 7)] = 4
        if with_type:
            # (1000, 1000) * 2 -> (1000, 1000, 2)
            ann_type = merged_ann_type
            full_ann = np.dstack([ann_inst, full_ann_type]) 
            merged_ann = np.dstack([ann_inst, merged_ann_type])
            full_ann = full_ann.astype("int32")
            merged_ann = merged_ann.astype("int32")
        else:
            full_ann = np.expand_dims(ann_inst, -1)
            full_ann = full_ann.astype("int32")
            merged_ann = np.expand_dims(ann_inst, -1)
            merged_ann = merged_ann.astype("int32")
        return full_ann, merged_ann

def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        # "kumar": lambda: __Kumar(),
        # "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name