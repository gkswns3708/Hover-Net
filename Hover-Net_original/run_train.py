"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
"""

import cv2

cv2.setNumThreads(0)
import argparse
import glob
import importlib
import inspect
import json
import os
import shutil

import matplotlib
import numpy as np
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from config import Config
from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine
from run_utils.utils import (
    check_log_dir,
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
)

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


####
class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""

    def __init__(self):
        super().__init__()
        return

    ####
    def view_dataset(self, mode="train"):
        """
        If running in headless mode, change plt.show() to plt.savefig()
        """
        self.nr_gpus = 1
        import matplotlib.pyplot as plt
        check_manual_seed(self.seed)

        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]

        dataloader = self._get_datagen(2, mode, target_info["gen"])
        
        for i, batch_data in enumerate(dataloader):  
            print(batch_data.keys(), "- batch_data")
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            
            # GUI 환경이면 plt.show(), 아니면 저장
            if os.environ.get("DISPLAY"):  # GUI 환경 확인
                plt.imshow(viz)
                plt.show()
            else:
                save_path = f"output_viz_{mode}_{i}.png"
                plt.imshow(viz)
                plt.savefig(save_path)
                print(f"Saved visualization: {save_path}")
            if i >= 10:
                break

        self.nr_gpus = -1
        return


    ####
    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0
        print(target_gen, "- target_gen")
        # ! Hard assumption on file type
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # to always ensure same input ordering

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode]
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """Simply run the defined run_step of the related method once."""
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        ####
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                prev_phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            print(f"net_name: {net_name}, net_info keys: {net_info.keys()}")

        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            # TODO: customize print-out for each run ?
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                print("Pretrained Path: ", pretrained_path)
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        net_state_dict = torch.load(pretrained_path)["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # * extremely slow to pass this on DGX with 1 GPU, why (?)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            # TODO: expand for external aug for scheduler
            nr_iter = opt["nr_epochs"] * len(loader_dict["train"])
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: standardize API for external hooks
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run(self):
        """Define multi-stage run or cross-validation or whatever in here."""
        self.nr_gpus = torch.cuda.device_count()
        print('Detect #GPUS: %d' % self.nr_gpus)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            
            # ✅ 수정된 부분: logs 디렉토리 내의 인덱스 폴더를 검색하여 새로운 경로 지정
            phase_idx = 0
            while True:
                save_path = os.path.join(self.log_dir, f"{phase_idx:02d}")
                if not os.path.exists(save_path):  # 새로운 폴더 이름이 존재하지 않을 때까지 증가
                    break
                phase_idx += 1
            print(save_path, "- save_path")
            rm_n_mkdir(save_path)  # ✅ 디렉토리 새로 생성 (중첩되지 않게)

            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path


####
if __name__ == "__main__":
    print(__doc__)
    args = docopt(__doc__, version="HoVer-Net v1.0")
    trainer = TrainManager()

    if args["--view"]:
        if args["--view"] != "train" and args["--view"] != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args["--view"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
        trainer.run()