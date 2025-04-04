import re
import glob
import os
import tqdm
import pathlib
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from misc.dataset import get_dataset

if __name__ == "__main__":
    # 클래스 레이블이 있으면 True
    type_classification = True
    
    win_size = [540, 540]
    step_size = [164, 164]
    # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
    extract_type = "mirror"  
    
    # 사용할 Open-Dataset
    # TODO: CoNSeP 데이터셋 외에 추가할 수 있음(Kumar, CPM17)
    dataset_name = "consep"
    save_root = "dataset/training_data/%s/" % dataset_name
    
    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".png", "dataset/CoNSeP/Train/Images/"),
            "ann": (".mat", "dataset/CoNSeP/Train/Labels/"),
        },
        "valid": {
            "img": (".png", "dataset/CoNSeP/Test/Images/"),
            "ann": (".mat", "dataset/CoNSeP/Test/Labels/"),
        },
    }
    
    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    
    for split_name, split_desc in dataset_info.items(): 
        img_ext, img_dir = split_desc['img']
        ann_ext, ann_dir = split_desc['ann']
        
        out_dir = f'{save_root}/{dataset_name}/{split_name}/{win_size[0]}x{win_size[1]}_{step_size[0]}x{step_size[1]}/'
        file_list = glob.glob(f'{ann_dir}/*{ann_ext}')
        file_list.sort()
        
        rm_n_mkdir(out_dir)
        
        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(total=len(file_list), bar_format=pbar_format, ascii=True, position=0)
        
        for file_idx, file_path in enumerate(file_list):
            #! stem: 파일명만 추출
            base_name = pathlib.Path(file_path).stem
            
            img = parser.load_img(f'{img_dir}/{base_name}{img_ext}')
            full_ann, merged_ann = parser.load_ann(f'{ann_dir}/{base_name}{ann_ext}', with_type=type_classification)
            # 여기서 나온 full_ann, merged_ann은 type_map과 inst_map을 합친 것.
            # img.shape = (1000, 1000, 3)
            # full_ann.shape = (1000, 1000, 2), merged_ann.shape = (1000, 1000, 2)
            # 그래서 모든 label이 포함된 버전과 아닌 버전이 있음.
            #! 0 ~ 2 : img
            #! 3 : full annotation 버전의 instance map
            #! 4 : full annotation 버전의 type map
            #! 5 : merged annotation 버전의 instance map
            #! 6 : merged annotation 버전의 type map
            img = np.concatenate([img, full_ann, merged_ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)
            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            
            pbarx.update()
        pbarx.close()