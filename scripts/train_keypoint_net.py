# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda选择
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import copy

from tensorboardX import SummaryWriter  # 这个替代wandb
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader
from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.evaluation.evaluate import evaluate_keypoint_net
from kp2d.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2d.utils.config import parse_train_file
from kp2d.utils.horovod import hvd_init, local_rank, rank, world_size, has_hvd
from kp2d.utils.logging import printcolor
from train_keypoint_net_utils import (_set_seeds, sample_to_cuda, fix_bn,
                                      setup_datasets_and_dataloaders)

try:
    import horovod.torch as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='KP2D training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), 'You need to provide a .ckpt of .yaml file'
    return args


def adjust_learning_rate(config, optimizer, epoch, decay=0.5, max_decays=10):
    """Sets the learning rate to the initial LR decayed by 0.5 every k epochs"""
    exponent = min(epoch // (config.model.scheduler.lr_epoch_divide_frequency / config.datasets.train.repeat),
                   max_decays)
    decay_factor = (config.model.scheduler.decay ** exponent)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['original_lr'] * decay_factor
        printcolor('Changing {} network learning rate to {:8.6f}'.format(param_group['name'], param_group['lr']),
                   'red')


def model_submodule(model):
    """Get submodule of the model in the case of DataParallel, otherwise return
    the model itself. """
    return model.module if hasattr(model, 'module') else model


def main(file):
    """
    KP2D training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Parse config
    config = parse_train_file(file)
    print(config)
    print(config.arch)

    if config.arch.seed is not None:
        _set_seeds(config.arch.seed)

    # Initialize horovod
    hvd_init()
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(n_threads)

    if world_size() > 1:
        printcolor('-' * 18 + 'DISTRIBUTED DATA PARALLEL ' + '-' * 18, 'cyan')
        device_id = local_rank()
        torch.cuda.set_device(device_id)
    else:
        printcolor('-' * 25 + 'SINGLE GPU ' + '-' * 25, 'cyan')

    # if config.arch.seed is not None:
    #     _set_seeds(config.arch.seed)

    if rank() == 0:
        printcolor('-' * 25 + ' MODEL PARAMS ' + '-' * 25)
        printcolor(config.model.params, 'red')

    # Setup model and datasets/dataloaders
    model = KeypointNetwithIOLoss(keypoint_net_learning_rate=config.model.optimizer.learning_rate,
                                  **config.model.params)

    # pre = torch.load(curPath + '/step1_model.ckpt')
    # pre = torch.load('/home/featurize/6times_model.ckpt')
    # model_dict = model.keypoint_net.state_dict()  # 模型参数
    # pretrained_dict = {k: v for k, v in pre['state_dict'].items() if k in model_dict}  # 选取名字一样的
    # model_dict.update(pretrained_dict)  # 更新一下。。
    # model.keypoint_net.load_state_dict(model_dict)  # 直接载入

    train_dataset, train_loader = setup_datasets_and_dataloaders(config.datasets, 50000)
    printcolor('({}) length: {}'.format("Train", len(train_dataset)))

    model = model.cuda()
    optimizer = optim.Adam(model.optim_params)
    if has_hvd():
        compression = hvd.Compression.none  # or hvd.Compression.fp16
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(),
                                             compression=compression)

        # Synchronize model weights from all ranks
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # checkpoint model
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    date_time = model_submodule(model).__class__.__name__ + '_' + date_time  # 模型名字加时间吧
    config.model.checkpoint_path = os.path.join(config.model.checkpoint_path, date_time)  # emm记录数据的文件所在地
    log_path = os.path.join(config.model.checkpoint_path, 'logs')
    os.makedirs(log_path, exist_ok=True)

    if rank() == 0:
        if not config.wandb.dry_run:
            summary = SummaryWriter(log_path)
        else:
            summary = None

        print('Saving models at {}'.format(config.model.checkpoint_path))
        os.makedirs(config.model.checkpoint_path, exist_ok=True)
    else:
        summary = None

    # Initial evaluation
    # evaluation(config, 0, model, summary, version=1)
    # Train
    for epoch in range(config.arch.epochs):
        printcolor('\n' + '-' * 50)
        if epoch == 0:
            printcolor('set hard data', 'red')
            train_loader.dataset.set_aug_type('hard')

        # train for one epoch (only log if eval to have aligned steps...)
        train(config, train_loader, model, optimizer, epoch, summary)

        # Model checkpointing, eval, and logging
        evaluation(config, epoch + 1, model, summary, version=1)

    printcolor('Training complete, models saved in {}'.format(config.model.checkpoint_path), "green")


def evaluation(config, completed_epoch, model, summary, version=1):
    # Set to eval mode
    model.eval()
    model.training = False

    use_color = config.model.params.use_color

    if rank() == 0:
        eval_shape = config.datasets.augmentation.image_shape[::-1]
        eval_params = [{'res': eval_shape, 'top_k': 300}]
        for params in eval_params:
            hp_dataset = PatchesDataset(root_dir=config.datasets.val.path, use_color=use_color,
                                        output_shape=params['res'], type='a')

            data_loader = DataLoader(hp_dataset,
                                     batch_size=1,
                                     pin_memory=False,
                                     shuffle=False,
                                     num_workers=0,
                                     worker_init_fn=None,
                                     sampler=None)
            print('Loaded {} image pairs '.format(len(data_loader)))

            printcolor('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']))

            printcolor('使用version ' + str(version))
            rep, loc, c1, c3, c5, mscore, cmn, mn, _, _, _, _, _, _, _, _ = \
                evaluate_keypoint_net(data_loader,
                                      model_submodule(model).keypoint_net,
                                      output_shape=params['res'],
                                      top_k=params['top_k'],
                                      use_color=use_color,
                                      version=version)

            if summary:
                size_str = str(params['res'][0]) + '_' + str(params['res'][1])
                summary.add_scalar('repeatability_' + size_str, rep, completed_epoch)
                summary.add_scalar('localization_' + size_str, loc, completed_epoch)
                summary.add_scalar('correctness_' + size_str + '_1', c1, completed_epoch)
                summary.add_scalar('correctness_' + size_str + '_3', c3, completed_epoch)
                summary.add_scalar('correctness_' + size_str + '_5', c5, completed_epoch)
                summary.add_scalar('mscore' + size_str, mscore, completed_epoch)

            printcolor('\n' + '-' * 50)
            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('Correctness d1 {:.3f}'.format(c1))
            print('Correctness d3 {:.3f}'.format(c3))
            print('Correctness d5 {:.3f}'.format(c5))
            print('MScore {:.3f}'.format(mscore))
            printcolor('-' * 50 + '\n')

    # Save checkpoint
    if config.model.save_checkpoint and rank() == 0:
        current_model_path = os.path.join(config.model.checkpoint_path, str(completed_epoch) + 'times_model.ckpt')
        printcolor('\nSaving model (epoch:{}) at {}'.format(completed_epoch, current_model_path), 'green')
        torch.save(
            {
                'state_dict': model_submodule(model_submodule(model).keypoint_net).state_dict(),
                'config': config
            }, current_model_path)


def train(config, train_loader, model, optimizer, epoch, summary):
    # Set to train mode
    model.train()
    if config.model.params.train_which_desc == 'netvlad':
        printcolor('锁定所有bn', 'red')
        model.keypoint_net.apply(fix_bn)  # fix batchnorm
        model.keypoint_net.netvlad.train()

    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    # if args.adjust_lr:
    adjust_learning_rate(config, optimizer, epoch)
    n_train_batches = len(train_loader)

    pbar = tqdm(enumerate(train_loader, 0),
                unit=' images',
                unit_scale=config.datasets.train.batch_size * world_size(),
                total=n_train_batches,
                smoothing=0,
                disable=(rank() > 0),
                ncols=225)  # 200差不多

    running_loss = running_recall = 0
    running_loss_total = {}

    train_progress = float(epoch) / float(config.arch.epochs)

    log_freq = 10

    random_batch = random.randint(0, n_train_batches - 1)

    for (now_batch, data) in pbar:

        # calculate loss
        optimizer.zero_grad()
        data_cuda = sample_to_cuda(data)
        loss, recall, loss_total, recall_total = model(data_cuda)  # 前向传播获取损失函数
        if has_hvd():
            loss = hvd.allreduce(loss.mean(), average=True, name='loss')

        # compute gradient
        loss.backward()

        # keep running data
        running_loss += float(loss)
        running_recall += recall
        for key in loss_total.keys():
            if key in running_loss_total.keys():
                running_loss_total[key] += float(loss_total[key])
            else:
                running_loss_total[key] = float(loss_total[key])

        # SGD step
        optimizer.step()

        # pretty progress bar
        if rank() == 0:
            # 服务器后台
            if config.model.params.train_which_desc == "netvlad":
                pbar.set_description(
                    ('{:<8d}: Train [ E {}, T {:d}, R {:.4f}, R_Avg {:.4f}, L {:.4f}, L_Avg {:.4f}'.format(
                        now_batch, epoch, epoch * config.datasets.train.repeat,
                        recall, running_recall / (now_batch + 1),
                        float(loss), float(running_loss) / (now_batch + 1))))
            else:
                pbar.set_description(
                    ('{:<8d}: Train [ E {}, T {:d}, R {:.4f}, R_Avg {:.4f}, L {:.4f}, L_Avg {:.4f}, '
                     'loc_loss {:.4f}, desc_loss {:.4f}, score_loss {:.4f}, score_addition_loss {:.4f}]'.format(
                        now_batch, epoch, epoch * config.datasets.train.repeat, recall,
                                          running_recall / (now_batch + 1),
                        float(loss), float(running_loss) / (now_batch + 1),
                        float(loss_total['loc_loss']), float(loss_total['desc_loss']),
                        float(loss_total['score_loss']), float(loss_total['score_addition_loss']))))

        now_batch += 1
        if now_batch % log_freq == 0:
            with torch.no_grad():
                if summary:
                    train_metrics = {
                        'train_progress': train_progress,
                        'running_loss': running_loss / (now_batch + 1),
                        'running_recall': running_recall / (now_batch + 1)
                    }
                    for key in running_loss_total.keys():
                        train_metrics['running_' + key] = running_loss_total[key] / (now_batch + 1)

                    for param_group in optimizer.param_groups:
                        train_metrics[param_group['name'] + '_learning_rate'] = param_group['lr']

                    for k, v in train_metrics.items():
                        summary.add_scalar(k, v, len(pbar) * epoch + now_batch - 1)

        if now_batch == random_batch:
            with torch.no_grad():
                if summary:
                    model(data_cuda, debug=True)
                    if len(model_submodule(model).vis) > 0:
                        for k, v in model_submodule(model).vis.items():
                            summary.add_image(k + 'epoch' + str(epoch), v)


if __name__ == '__main__':
    # args = parse_args()
    # main(args.file)
    file_path = rootPath + '/kp2d/configs/v2_preact_resnet.yaml'
    main(file_path)
