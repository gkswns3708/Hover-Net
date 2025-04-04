import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import trainloader
from models.base_hovernet import targets, net_desc, opt



import os
import argparse
import importlib
from glob import glob
from tqdm import tqdm

from config import normal_Config, uniform_Config

def worker_init_fn(worker_id):
    # ! Multiprocessing 환경에서는 각 worker가 독립적인 상태를 유지하려고 함. 
    # ! Pytorch의 DDP를 따를 경우에는 Mother Process에서부터 Spawn 방식을 통해 inital Random seed를 부여 받게 되는데
    # ! 이 경우 각 worker가 같은 Random seed를 가지게 되어, 동일한 데이터를 처리하게 되거나, 동일한 augmentation을 적용될 수 있음.
    # ! 이를 방지하기 위해 각 worker에게 다른 Random seed를 부여해주는 것이 좋음.
    # ! 아래의 코드는 각 worker에게 다른 Random seed를 부여하는 코드임.
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.randint(0, 2 ** 32, (1, ))[0].cpu().item() + worker_id
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    return 

def main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # Load dataset
    print("Loading dataset...")

    run_mode  = config['run_mode']
    input_dataset = trainloader.FileLoader(
        glob(os.path.join(config['dataset_path'], '*.npy')),     
        mode=config['run_mode'],
        with_type=config['with_type'],
        setup_augmentor=True, # 이거 True / False 차이 알아보기
        target_gen=[targets.gen_targets, {}],
        **config['shape_info'][config['run_mode']]
    )
    dataloader = DataLoader(
        input_dataset,
        num_workers=2,
        batch_size=normal_Config['batch_size'],
        shuffle=normal_Config['run_mode'] == "train",
        drop_last=normal_Config['run_mode'] == "train",
        # worker_init_fn=worker_init_fn, # TODO: multiGPU에서 
    )
    
    phase_list = opt.get_config(
        config['num_types'], 
        config['sampling_mode']
    )['phase_list']
    # Model Load
    for phase_idx, phase_info in enumerate(phase_list):
        run_once(phase_info, engine_opt, svae_p)
        
    # Loss function
    
    
    
    # ~~ 7
    # ~~ 7
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hover-Net')
    parser.add_argument('sampling_mode', type=str, choices=['normal', 'uniform'], help='Sampling mode')
    args = parser.parse_args()
    
    
    config = normal_Config if args.sampling_mode == 'normal' else uniform_Config
    main()

    