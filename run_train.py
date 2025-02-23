
import torch




















def worker_init_fn(worker_id):
    # ! Multiprocessing 환경에서는 각 worker가 독립적인 상태를 유지하려고 함. 
    # ! Pytorch의 DDP를 따를 경우에는 Mother Process에서부터 Spawn 방식을 통해 inital Random seed를 부여 받게 되는데
    # ! 이 경우 각 worker가 같은 Random seed를 가지게 되어, 동일한 데이터를 처리하게 되거나, 동일한 augmentation을 적용될 수 있음.
    # ! 이를 방지하기 위해 각 worker에게 다른 Random seed를 부여해주는 것이 좋음.
    # ! 아래의 코드는 각 worker에게 다른 Random seed를 부여하는 코드임.
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.randint(0, 2 ** 32, (1, ))[0].cpu().item() + worker_id
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return 