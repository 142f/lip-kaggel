import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import AVLip


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler




def create_dataloader(opt):
    """
    创建一个健壮的 DataLoader。
    - 修复了 sampler 和 shuffle 的冲突。
    - 强制 num_workers=0 以避免在 Notebook 环境中死锁。
    - 保留了类别平衡采样的功能。
    """
    print("Initializing dataset...")
    dataset = AVLip(opt)

    sampler = None
    # 只有在训练模式且启用了 class_bal 时，才创建 sampler
    if opt.isTrain and opt.class_bal:
        print("Class balancing is enabled. Creating a balanced sampler.")
        sampler = get_bal_sampler(dataset)

    # 关键逻辑：如果 sampler 被创建了，shuffle 必须为 False。
    # 否则，根据 opt 的配置决定是否 shuffle。
    if sampler is not None:
        use_shuffle = False
        print("Sampler is in use, setting shuffle=False.")
    else:
        use_shuffle = not opt.serial_batches if opt.isTrain else False
        print(f"Sampler is NOT in use, setting shuffle={use_shuffle}.")

    print(f"Forcing num_workers to 0 to prevent potential deadlocks.")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=use_shuffle,  # 使用我们计算出的正确 shuffle 状态
        sampler=sampler,
        num_workers=0,        # 强制设为 0
    )

    print("DataLoader created successfully.")
    return data_loader