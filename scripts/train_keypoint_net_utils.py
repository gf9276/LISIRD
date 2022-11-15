# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset, DataLoader

from kp2d.datasets.augmentations import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample)
from kp2d.datasets.coco import COCOLoader
from kp2d.utils.horovod import rank, world_size
import matplotlib.pyplot as plt


def fix_bn(m):
    # 锁定bn用的，，，
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def sample_to_cuda(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_cuda(data[key])
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_cuda(key))
        return data_cuda
    else:
        return data.to('cuda')


def image_transforms(shape, jittering):
    def train_transforms(sample, aug_type):
        sample = spatial_augment_sample(sample)  # 先转了再说
        sample = resize_sample(sample, image_shape=shape)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_parameters=jittering, aug_type=aug_type)
        return sample

    # 返回两个难度的吧
    return {'train': train_transforms}


def _set_seeds(seed=42):
    """Set Python random seeding and PyTorch seeds.
    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_datasets_and_dataloaders(config, data_size=None):
    """Prepare datasets for training, validation and test."""

    def _worker_init_fn(worker_id):
        """Worker init fn to fix the seed of the workers"""
        # seed = 43 + worker_id
        seed = torch.initial_seed() % 2 ** 32 + worker_id  # worker_id 可以不加，每个epoch都不一样，**优先级很高的
        np.random.seed(seed)
        random.seed(seed)
        # print(str(torch.initial_seed()) + '\n')
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

    data_transforms = image_transforms(shape=config.augmentation.image_shape, jittering=config.augmentation.jittering)
    train_dataset = COCOLoader(config.train.path, data_transform=data_transforms['train'], data_size=data_size)
    # Concatenate dataset to produce a larger one
    if config.train.repeat > 1:
        train_dataset = ConcatDataset([train_dataset for _ in range(config.train.repeat)])

    # Create loaders
    if world_size() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size(), rank=rank())
    else:
        sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              pin_memory=True,
                              shuffle=not (world_size() > 1),
                              num_workers=config.train.num_workers,
                              worker_init_fn=_worker_init_fn,
                              sampler=sampler,
                              drop_last=True)
    return train_dataset, train_loader
