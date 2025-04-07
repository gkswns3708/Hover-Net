import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


import os
import argparse
import importlib
from glob import glob
from tqdm import tqdm
from termcolor import colored
from collections import OrderedDict

from dataloader import trainloader
from config import normal_Config, uniform_Config

from models.base_hovernet import targets
from models.base_hovernet.utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss
from models.base_hovernet.opt import get_config

# ✅ 모델 초기화 개선 함수 (He Initialization 사용)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

def load_resnet(model, pretrained_path, device="cuda"):
    """
    ResNet 모델을 pretrained_path에서 로드하고 모델에 추가하는 함수
    Args:
        model (torch.nn.Module): 모델 객체
        pretrained_path (str): pretrained ResNet 경로
        device (str): 학습할 장치 (기본값: "cuda")
    Returns:
        model (torch.nn.Module): ResNet을 로드한 모델
    """
    assert os.path.isfile(pretrained_path), f"ResNet 경로가 잘못되었습니다: {pretrained_path}"
    
    net_state_dict = torch.load(pretrained_path, map_location=device)
    
    # ResNet의 state_dict만 가져오기
    resnet_state_dict = {k: v for k, v in net_state_dict.items() if "resnet" in k}
    
    # 모델에 해당 state_dict 로드하기
    load_feedback = model.load_state_dict(resnet_state_dict, strict=False)
    
    print(f"✅ Loaded ResNet from: {pretrained_path}")
    print("Missing Variables: ", load_feedback.missing_keys)
    print("Unexpected Variables: ", load_feedback.unexpected_keys)
    
    return model

def load_pretrained_model(model, pretrained_path, optimizer=None, scheduler=None, device="cuda"):
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 모델 로드
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # 옵티마이저와 스케줄러도 함께 로드 (선택 사항)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Loaded Pretrained Model from: {pretrained_path}")
    print(f"Last Saved Epoch: {checkpoint['epoch']}")
    print(f"Last Saved Loss: {checkpoint['loss']:.4f}")
    
    return model, optimizer, scheduler

def save_model(model, optimizer, scheduler, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"✅ Model saved to {save_path}")



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

def validate(model, valid_loader, loss_opts, loss_func_dict, device="cuda"):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    num_batches = 0

    with torch.no_grad():  # 검증 중에는 Gradient 계산을 하지 않음
        pbar = tqdm(valid_loader, desc="Validation")
        
        for batch in pbar:
            imgs = batch["img"].permute(0, 3, 1, 2).to(device)
            true_np = batch["np_map"].to(device)
            true_hv = batch["hv_map"].to(device)
            true_tp = batch.get("tp_map", None)
            if true_tp is not None:
                true_tp = true_tp.to(device)

            pred_dict = model(imgs)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = torch.softmax(pred_dict["np"], dim=-1)
            if "tp" in pred_dict:
                pred_dict["tp"] = torch.softmax(pred_dict["tp"], dim=-1)

            loss = 0
            for branch_name, losses in loss_opts.items():
                for loss_name, weight in losses.items():
                    loss_func = loss_func_dict[loss_name]

                    if branch_name == "np":
                        pred_np = pred_dict["np"][..., 1]
                        true_np = true_np.to(torch.int64)
                        true_np_onehot = F.one_hot(true_np, num_classes=2).type(torch.float32)
                        loss += weight * loss_func(pred_np, true_np_onehot[..., 1])

                    elif branch_name == "hv":
                        pred_hv = pred_dict["hv"]
                        if loss_name == "msge":
                            focus = true_np_onehot[..., 1]
                            loss += weight * loss_func(pred_hv, true_hv, focus)
                        else:
                            loss += weight * loss_func(pred_hv, true_hv)

                    elif branch_name == "tp" and true_tp is not None:
                        pred_tp = pred_dict["tp"].permute(0, 3, 1, 2)
                        if not true_tp.dtype == torch.int64:
                            true_tp = true_tp.to(torch.int64)

                        true_tp_onehot = F.one_hot(true_tp, num_classes=pred_tp.shape[1])
                        true_tp_onehot = true_tp_onehot.permute(0, 3, 1, 2).type(torch.float32)
                        loss += weight * loss_func(pred_tp, true_tp_onehot)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches
    return avg_loss

def train_phase(config, phase_idx, model=None, model_path=None, device="cuda"):
    phase_info = config["phase_list"][phase_idx]
    run_info = phase_info["run_info"]
    net_info = run_info["net"]
    pretrained_path = net_info["pretrained"]
    
    # 모델 초기화 (Phase 1이면 모델 새로 생성, Phase 2면 기존 모델 이어받기)
    if model is None:
        model = net_info["desc"]().to(device)
        optimizer_class, optimizer_params = net_info["optimizer"]
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = net_info["lr_scheduler"](optimizer)

        if pretrained_path == -1:
            print("✅ Using He Initialization (No Pretrained Model Loaded)")
            model.apply(initialize_weights)
        else:
            # ✅ ResNet 모델 로드 상황에서는 load_resnet() 호출하도록 변경
            print(f"✅ Loading Pretrained ResNet from: {pretrained_path}")
            model = load_resnet(model, pretrained_path, device)  # 여기 수정됨!
    else:
        print(f"✅ Loading Phase 1 Model from: {model_path}")
        optimizer_class, optimizer_params = net_info["optimizer"]
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = net_info["lr_scheduler"](optimizer)
        
        # ✅ Phase 1 모델 로드
        checkpoint_path = os.path.join(model_path, "phase1_model.tar")
        model, optimizer, scheduler = load_pretrained_model(
            model, checkpoint_path, optimizer, scheduler, device
        )

    loss_opts = net_info["extra_info"]["loss"]
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }

    train_dataset = trainloader.FileLoader(
        glob(os.path.join(normal_Config['train_dataset_path'], '*.npy')),     
        mode=normal_Config['run_mode'],
        with_type=normal_Config['with_type'],
        setup_augmentor=True, # 이거 True / False 차이 알아보기
        target_gen=[targets.gen_targets, {}],
        **normal_Config['shape_info'][normal_Config['run_mode']]
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=2,
        batch_size=normal_Config['batch_size'],
        shuffle=normal_Config['run_mode'] == "train",
        drop_last=normal_Config['run_mode'] == "train",
        # worker_init_fn=worker_init_fn, # TODO: multiGPU에서 
    )
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    for key, value in train_dataset[0].items():
        print(f"{key}: {value.shape}")
    
    valid_dataset = trainloader.FileLoader(
        glob(os.path.join(normal_Config['valid_dataset_path'], '*.npy')),     
        mode=normal_Config['run_mode'],
        with_type=normal_Config['with_type'],
        setup_augmentor=True, # 이거 True / False 차이 알아보기
        target_gen=[targets.gen_targets, {}],
        **normal_Config['shape_info'][normal_Config['run_mode']]
    )
    valid_loader = DataLoader(
        valid_dataset,
        num_workers=2,
        batch_size=normal_Config['batch_size'],
        shuffle=normal_Config['run_mode'] == "train",
        drop_last=normal_Config['run_mode'] == "train",
        # worker_init_fn=worker_init_fn, # TODO: multiGPU에서 
    )

    nr_epochs = phase_info["nr_epochs"]
    for epoch in range(nr_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            imgs = batch["img"].permute(0, 3, 1, 2).to(device)
            true_np = batch["np_map"].to(device)
            true_hv = batch["hv_map"].to(device)
            true_tp = batch.get("tp_map", None)
            if true_tp is not None:
                true_tp = true_tp.to(device)

            optimizer.zero_grad()
            pred_dict = model(imgs)
            pred_dict = {k: v.permute(0, 2, 3, 1) for k, v in pred_dict.items()}

            loss = 0

            # 손실값들을 딕셔너리로 저장
            loss_values = {
                "np_bce": 0.0,
                "np_dice": 0.0,
                "hv_mse_x": 0.0,
                "hv_mse_y": 0.0,
                "hv_msge": 0.0,
                "tp_bce": 0.0,
                "tp_dice": 0.0,
            }

            for branch_name, losses in loss_opts.items():
                for loss_name, weight in losses.items():
                    loss_func = loss_func_dict[loss_name]

                    if branch_name == "np":
                        pred_np = torch.softmax(pred_dict["np"], dim=-1)
                        true_np = true_np.to(torch.int64)
                        true_np_onehot = F.one_hot(true_np, num_classes=2).type(torch.float32)
                        
                        if loss_name == "bce":
                            np_loss = weight * loss_func(pred_np, true_np_onehot)
                            loss_values["np_bce"] = np_loss.item()
                        elif loss_name == "dice":
                            np_loss = weight * loss_func(true_np_onehot, pred_np)
                            loss_values["np_dice"] = np_loss.item()
                        
                        loss += np_loss

                    elif branch_name == "hv":
                        pred_hv = pred_dict["hv"]
                        true_hv_x = true_hv[..., 0]
                        true_hv_y = true_hv[..., 1]
                        pred_hv_x = pred_hv[..., 0]
                        pred_hv_y = pred_hv[..., 1]

                        if loss_name == "msge":
                            focus = true_np_onehot[..., 1]
                            hv_loss = weight * loss_func(true_hv, pred_hv, focus)
                            loss_values["hv_msge"] = hv_loss.item()
                        else:  # mse loss
                            hv_loss_x = weight * loss_func(pred_hv_x, true_hv_x)
                            hv_loss_y = weight * loss_func(pred_hv_y, true_hv_y)
                            hv_loss = (hv_loss_x + hv_loss_y) / 2
                            
                            loss_values["hv_mse_x"] = hv_loss_x.item()
                            loss_values["hv_mse_y"] = hv_loss_y.item()
                        
                        loss += hv_loss

                    elif branch_name == "tp":
                        pred_tp = torch.softmax(pred_dict["tp"], dim=-1)
                        true_tp = true_tp.to(torch.int64)
                        true_tp_onehot = F.one_hot(true_tp, num_classes=8).type(torch.float32)
                        
                        if loss_name == "bce":
                            tp_loss = weight * loss_func(pred_tp, true_tp_onehot)
                            loss_values["tp_bce"] = tp_loss.item()
                        elif loss_name == "dice":
                            tp_loss = weight * loss_func(true_tp_onehot, pred_tp)
                            loss_values["tp_dice"] = tp_loss.item()
                        
                        loss += tp_loss

            # ✅ 손실값들을 한 줄로 출력하기
            print(f"np_bce: {loss_values['np_bce']:.4f}, np_dice: {loss_values['np_dice']:.4f}, "
                  f"hv_mse_x: {loss_values['hv_mse_x']:.4f}, hv_mse_y: {loss_values['hv_mse_y']:.4f}, hv_msge: {loss_values['hv_msge']:.4f}, "
                  f"tp_bce: {loss_values['tp_bce']:.4f}, tp_dice: {loss_values['tp_dice']:.4f}")
            # Backpropagation
            loss.backward()
            
            # ✅ Gradient Clipping 적용
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{nr_epochs}], Loss: {avg_loss:.4f}")

    # ✅ Phase 1 모델을 저장하기
    if model_path is not None and phase_idx == 0:
        save_model(model, optimizer, scheduler, epoch, avg_loss, os.path.join(model_path, "phase1_model.tar"))

    # ✅ Phase 2 모델을 저장하기
    if model_path is not None and phase_idx == 1:
        save_model(model, optimizer, scheduler, epoch, avg_loss, os.path.join(model_path, "phase2_model.tar"))

    return model

def run_training():
    config = get_config(8, 'original')

    # Phase 1 학습 (모델 새로 생성)
    model = train_phase(config, phase_idx=0, model_path="./checkpoints")
    
    # Phase 2 학습 (Phase 1 모델을 이어받기)
    model = train_phase(config, phase_idx=1, model=model, model_path="./checkpoints")

def main(config):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])    
    # hyperperamters
    nr_type = config['nr_type']
    mode = config['model_mode']
    run_training()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hover-Net')
    parser.add_argument('--sampling_mode', type=str, choices=['normal', 'uniform'], help='Sampling mode')
    args = parser.parse_args()
    
    
    config = normal_Config if args.sampling_mode == 'normal' else uniform_Config
    main(config)

    