# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""Default kp2d configuration parameters (overridable in configs/*.yaml)
"""

import os
from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''  # Run name
cfg.debug = True  # Debugging flag
########################################################################################################################
### ARCH  我不知道为啥要交arch hhh
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 1244  # Random seed for Pytorch/Numpy initialization 4,294,967,295 上限
cfg.arch.epochs = 50  # Maximum number of epochs  轮次
########################################################################################################################
### WANDB  我不用这个
########################################################################################################################
cfg.wandb = CN()
cfg.wandb.dry_run = True  # Wandb dry-run (not logging)
cfg.wandb.name = ''  # Wandb run name
cfg.wandb.project = os.environ.get("WANDB_PROJECT", "")  # Wandb project
cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")  # Wandb entity
cfg.wandb.tags = []  # Wandb tags
cfg.wandb.dir = ''  # Wandb save folder
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
# cfg.model.checkpoint_path = r'D:\data\experiments\kp2d'              # Checkpoint path for model saving
cfg.model.checkpoint_path = '/home/featurize/data/experiments/kp2d/'  # Checkpoint path for model saving
cfg.model.save_checkpoint = True
########################################################################################################################
### MODEL.SCHEDULER
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.decay = 0.5  # Scheduler decay rate
cfg.model.scheduler.lr_epoch_divide_frequency = 2  # Schedule number of epochs when to decay the initial learning rate by decay rate
########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.learning_rate = 0.001
cfg.model.optimizer.weight_decay = 0.0
########################################################################################################################
### MODEL.PARAMS
########################################################################################################################
cfg.model.params = CN()
cfg.model.params.keypoint_loss_weight = 1.0  # Keypoint loss weight
cfg.model.params.descriptor_loss_weight = 1.0  # Descriptor loss weight
cfg.model.params.score_loss_weight = 1.0  # Score loss weight
cfg.model.params.use_color = True  # Use color or grayscale images
cfg.model.params.do_upsample = True  # Upsample descriptors
cfg.model.params.do_cross = True  # Use cross-border keypoints
cfg.model.params.descriptor_loss = True  # Use hardest negative mining descriptor loss
cfg.model.params.use_new_descriptor_loss = True  # 使用新的Loss description
cfg.model.params.add_score_addition_loss = True  # 增加额外的分数损失函数
cfg.model.params.add_desc_relate_score_loss = True  # 增加损失函数关联分数
cfg.model.params.keypoint_net_type = 'KeypointNet'  # 'KeypointNet', 'KeypointResnet',
cfg.model.params.train_which_desc = ''
########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CN()
########################################################################################################################
### DATASETS.AUGMENTATION
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = (240, 320)  # Image shape
cfg.datasets.augmentation.jittering = (0.5, 0.5, 0.2, 0.05)  # Color jittering values
########################################################################################################################
### DATASETS.TRAIN
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 8  # Training batch size
cfg.datasets.train.num_workers = 6  # Training number of workers
cfg.datasets.train.path = '/home/featurize/data/coco2017/train2017/'  # Training data path (COCO dataset)
# cfg.datasets.train.path = r'D:\coco2017\train2017'    # Training data path (COCO dataset)
cfg.datasets.train.repeat = 1  # Number of times training dataset is repeated per epoch
########################################################################################################################
### DATASETS.VAL
########################################################################################################################
cfg.datasets.val = CN()
cfg.datasets.val.path = '/home/featurize/data/hpatches-sequences-release/'  # Validation data path (HPatches)
# cfg.datasets.val.path = r'D:\hpatches-sequences-release'
########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''  # Run configuration file
cfg.default = ''  # Run default configuration file
cfg.wandb.url = ''  # Wandb URL


########################################################################################################################

def get_cfg_defaults():
    return cfg.clone()
