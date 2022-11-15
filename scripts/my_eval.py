import argparse
import os
from datetime import datetime
import sys
import time
import re
from math import pi

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kp2d.utils.image import to_color_normalized
from kp2d.evaluation.descriptor_evaluation import select_k_best
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.networks.keypoint_preact_resnet import KeypointPreactResnet
from kp2d.evaluation.evaluate import evaluate_keypoint_net
from kp2d.utils.keypoints import warp_keypoints, get_uv_norm, warp_homography_batch, get_uv_unnorm, get_border_mask
from kp2d.datasets.augmentations import tensorimg2cv2, cv22tensorimg, my_show_img, get_fixed_h
from skimage import feature, exposure
import matplotlib.pyplot as plt
import copy


def main(model_file):
    # 获取预训练模型，里面有两部分 'state_dict'和'config'
    pretrained_model = model_file

    # 下载模型，以字典的形式
    checkpoint = torch.load(pretrained_model)

    # 获取模型参数
    model_args = checkpoint['config'].model.params
    config = checkpoint['config']

    # 获取模型类型
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        keypoint_net_type = checkpoint['config']['model']['params']['keypoint_net_type']
    else:
        keypoint_net_type = 'KeypointNet'

    # 创建模型，复制keypointwithio那一部分就可以了
    if keypoint_net_type == 'KeypointNet':
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                   do_upsample=model_args['do_upsample'],
                                   do_cross=model_args['do_cross'])
    elif keypoint_net_type == 'KeypointResnet':
        keypoint_net = KeypointResnet()
    elif keypoint_net_type == 'KeypointPreactResnet':
        keypoint_net = KeypointPreactResnet()
    else:
        raise NotImplemented('Keypoint net type not supported {}'.format(keypoint_net_type))

    # 载入参数
    # keypoint_net.load_state_dict(checkpoint['state_dict'])
    pre = checkpoint
    model_dict = keypoint_net.state_dict()  # 模型参数
    pretrained_dict = {k: v for k, v in pre['state_dict'].items() if k in model_dict}  # 选取名字一样的
    model_dict.update(pretrained_dict)  # 更新一下。。
    keypoint_net.load_state_dict(model_dict)  # 直接载入
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()  # ！！！！！！别忘了

    print('Loaded KeypointNet from {}'.format(pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    eval_shape = config.datasets.augmentation.image_shape[::-1]
    eval_params = [
        {'res': eval_shape, 'top_k': 1200, 'des_dis': 9999, 'score_th': 0.5, 'multiscale': 1, 'draw_corr': True,
         'sift': False},
        # {'res': eval_shape, 'top_k': 300, 'des_dis': 9999, 'score_th': 0.5, 'multiscale': 2, 'draw_corr': True,
        #  'sift': False},
        # {'res': eval_shape, 'top_k': 300, 'des_dis': 3, 'score_th': 0.5, 'multiscale': 3, 'draw_corr': True,
        #  'sift': False},
        # {'res': eval_shape, 'top_k': 300, 'des_dis': 3, 'score_th': 0.5, 'multiscale': 4, 'draw_corr': True,
        #  'sift': False},
        # {'res': eval_shape, 'top_k': 300, 'des_dis': 3, 'score_th': 0.5, 'multiscale': 5, 'draw_corr': True,
        #  'sift': False},
        # {'res': eval_shape, 'top_k': 300, 'des_dis': 3, 'score_th': 0.5, 'multiscale': 6, 'draw_corr': True,
        #  'sift': False},
        # {'res': eval_shape, 'top_k': 300, 'des_dis': 3, 'score_th': 0.5, 'multiscale': 9, 'draw_corr': True,
        #  'sift': False},
    ]

    for params in eval_params:
        matches_nbr_sum = 0  # 所有图片配对点数和
        correct_matches_nbr_sum = 0  # 上述变量加上"正确的"的条件 hhh
        correctness1 = []
        correctness3 = []
        correctness5 = []
        detect_time = 0

        tmp_name = re.sub("[^\w]", " ", str(params)).split()
        tmp_name = '_'.join(tmp_name)
        catch_point_dir = '/home/featurize/data/experiments/kp2d/' + \
                          os.path.split(config.model.checkpoint_path)[-1] + '/cp_' + tmp_name + '/'
        # catch_point_dir = '/home/featurize/data/experiments/kp2d/haha' + '/cp' + tmp_name + '/'
        os.makedirs(catch_point_dir, exist_ok=True)  # 创建路径

        # hp_dataset = PatchesDataset(root_dir=config.datasets.val.path, use_color=True,
        #                             output_shape=params['res'], type='a')  # 获取数据集
        hp_dataset = PatchesDataset(root_dir='/home/featurize/data/hpatches-sequences-release/', use_color=True,
                                    output_shape=params['res'], type='a')  # 获取数据集

        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,  # 一定要是1，要问为什么，因为我没写B>1的
                                 pin_memory=False,
                                 shuffle=False,  # 打乱个锤子
                                 num_workers=0,  # 不是0没法调试。。。我也不知道为什么，和训练过程的不一样
                                 worker_init_fn=None,  # 不需要seed
                                 sampler=None)

        # 先统一做个评估，后面再分开来一次。。。
        rep, loc, c1, c3, c5, mscore, cmn, mn, \
        rep_l, loc_, c1_1, c3_l, c5_l, mscore_l, cmn_l, mn_l = evaluate_keypoint_net(data_loader,
                                                                                     keypoint_net,
                                                                                     output_shape=params['res'],
                                                                                     top_k=params['top_k'],
                                                                                     use_color=True,
                                                                                     version=params['multiscale'])

        with open(catch_point_dir + '000nbr.txt', 'a') as f:
            f.write('Repeatability {0:.3f}\n'.format(rep))
            f.write('Localization Error {0:.3f}\n'.format(loc))
            f.write('Correctness d1 {:.3f}\n'.format(c1))
            f.write('Correctness d3 {:.3f}\n'.format(c3))
            f.write('Correctness d5 {:.3f}\n'.format(c5))
            f.write('MScore {:.3f}\n'.format(mscore))
            f.write('correct_matches_nbr {:.4f}\n'.format(cmn))
            f.write('matches_nbr {:.4f}\n'.format(mn))

        keypoint_net.eval()
        keypoint_net.training = False
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                # 输入图片的batch_size必须为1，毕竟我这是评估，又不是训练

                if i == 103:
                    print('hehe')

                imgs = []
                warped_imgs = []
                homographys = []
                warped_homographys = []
                r_and_theta = [[1, 0], [0.6, 0], [0.7, pi / 4], [0.7, -pi / 4]]

                for param in r_and_theta:
                    tmp1, tmp2, tmp3 = get_fixed_h([256, 320], param[0], param[1], sample['image'])
                    imgs.append(tmp2)
                    homographys.append(tmp3)
                    tmp1, tmp2, tmp3 = get_fixed_h([256, 320], param[0], param[1], sample['warped_image'])
                    warped_imgs.append(tmp2)
                    warped_homographys.append(tmp3)
                imgs = torch.cat(imgs, 0).cuda()
                homographys = torch.cat(homographys, 0).cuda()
                warped_imgs = torch.cat(warped_imgs, 0).cuda()
                warped_homographys = torch.cat(warped_homographys, 0).cuda()
                imgsg = to_color_normalized(imgs.clone())
                warped_imgsg = to_color_normalized(warped_imgs.clone())

                image = to_color_normalized(sample['image'].cuda())  # 获取原始图片
                warped_image = to_color_normalized(sample['warped_image'].cuda())  # 获取扭曲图片

                # 对图像进行处理，恢复cv2格式
                img1 = tensorimg2cv2(sample['image'])
                img2 = tensorimg2cv2(sample['warped_image'])

                if params['sift']:
                    version = params['multiscale']
                    start_time = time.time()
                    conf_threshold = params['score_th']
                    sift = cv2.SIFT_create()
                    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
                    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
                    end_time = time.time()
                    detect_time += end_time - start_time
                    keypoints = np.array([kp1[tmp].pt for tmp in range(len(kp1))])
                    desc = des1
                    warped_keypoints = np.array([kp2[tmp].pt for tmp in range(len(kp2))])
                    warped_desc = des2
                else:
                    version = params['multiscale']
                    start_time = time.time()
                    conf_threshold = params['score_th']

                    if version == 1:
                        score1, coord1, desc1, netvlad1 = keypoint_net(imgsg, is_need_netvlad=True)
                        score2, coord2, desc2, netvlad2 = keypoint_net(warped_imgsg, is_need_netvlad=True)
                        device = score1.device
                        d_mat = torch.mm(netvlad1, netvlad2.t())
                        d_mat_cao = d_mat.clone()
                        d_mat = torch.sqrt(2 - 2 * torch.clamp(d_mat, min=-1, max=1))  # 转化成欧式距离
                        tmp = d_mat.argmin()  # 输出居然是先行的，比如说 6 就是 第二行第零列
                        line = tmp // 4
                        raw = tmp % 4
                        if d_mat[line, raw] >= torch.diag(d_mat).min():
                            line = 0
                            raw = 0
                        H = 256
                        W = 320
                        if i == 66:
                            print('cao')
                        coord1_norm = get_uv_norm(coord1, H, W)
                        coord2_norm = get_uv_norm(coord2, H, W)
                        coord1_norm = warp_homography_batch(coord1_norm, homographys)
                        coord2_norm = warp_homography_batch(coord2_norm, warped_homographys)
                        coord1 = get_uv_unnorm(coord1_norm, H, W)
                        coord2 = get_uv_unnorm(coord2_norm, H, W)

                        B, _, Hc, Wc = score1.shape

                        border_mask_ori = get_border_mask(B, Hc, Wc, device)
                        border_mask_ori = border_mask_ori.gt(1e-3).to(device)  # return True or False
                        # out-of-border(oob) mask source-to-target(s2t)
                        oob_mask1 = coord1_norm[:, :, :, 0].lt(1) & coord1_norm[:, :, :, 0].gt(-1) & \
                                    coord1_norm[:, :, :, 1].lt(1) & coord1_norm[:, :, :, 1].gt(-1)
                        border_mask1 = border_mask_ori & oob_mask1  # used in source img network output
                        # out-of-border(oob) mask source-to-target(s2t)
                        oob_mask2 = coord2_norm[:, :, :, 0].lt(1) & coord2_norm[:, :, :, 0].gt(-1) & \
                                    coord2_norm[:, :, :, 1].lt(1) & coord2_norm[:, :, :, 1].gt(-1)
                        border_mask2 = border_mask_ori & oob_mask2  # used in source img network output
                        score1 = score1 * border_mask1.float().unsqueeze(1)
                        score2 = score2 * border_mask2.float().unsqueeze(1)

                        if line == 0 and raw == 0:
                            score1 = score1[line].unsqueeze(0)
                            coord1 = coord1[line].unsqueeze(0)
                            desc1 = desc1[line].unsqueeze(0)

                            score2 = score2[raw].unsqueeze(0)
                            coord2 = coord2[raw].unsqueeze(0)
                            desc2 = desc2[raw].unsqueeze(0)
                        else:
                            factor = torch.cat([d_mat_cao[0, 0].unsqueeze(0), d_mat_cao[line, raw].unsqueeze(0)], dim=0)
                            factor = torch.softmax(factor, dim=0)

                            nbr1, _ = score1[0].view(-1).topk(int(params['top_k'] * factor[0]))
                            nbr1 = nbr1[-1]
                            score1[0][score1[0] >= nbr1] = 1
                            score1[0][score1[0] < nbr1] = 0

                            nbr2, _ = score2[0].view(-1).topk(int(params['top_k'] * factor[0]))
                            nbr2 = nbr2[-1]
                            score2[0][score2[0] >= nbr2] = 1
                            score2[0][score2[0] < nbr2] = 0

                            score1 = torch.cat([score1[line].unsqueeze(0), score1[0].unsqueeze(0)], dim=3)
                            coord1 = torch.cat([coord1[line].unsqueeze(0), coord1[0].unsqueeze(0)], dim=3)
                            desc1 = torch.cat([desc1[line].unsqueeze(0), 100 * desc1[0].unsqueeze(0)], dim=3)

                            score2 = torch.cat([score2[raw].unsqueeze(0), score2[0].unsqueeze(0)], dim=3)
                            coord2 = torch.cat([coord2[raw].unsqueeze(0), coord2[0].unsqueeze(0)], dim=3)
                            desc2 = torch.cat([desc2[raw].unsqueeze(0), 100 * desc2[0].unsqueeze(0)], dim=3)


                    elif version == 2:
                        score1, coord1, desc1 = keypoint_net(image)
                        score2, coord2, desc2 = keypoint_net(warped_image)
                    elif version == 3:
                        coord1, score1, desc1 = keypoint_net(image, ['scale'])
                        coord2, score2, desc2 = keypoint_net(warped_image, ['scale'])
                    elif version == 4:
                        coord1, score1, desc1_dict = keypoint_net(image, ['perspective', 'rotate'])
                        coord2, score2, desc2_dict = keypoint_net(warped_image, ['perspective', 'rotate'])

                        dist_mat = {}
                        dist_mat_mean = {}  # 这是个占比，越大越好
                        suit_key = None
                        now_dist = 0
                        for key in desc1_dict.keys():
                            B, C, Hc, Wc = desc1_dict[key].shape
                            dist_mat[key] = torch.mm(desc1_dict[key].view(C, -1).t(), desc2_dict[key].view(C, -1))
                            dist_mat[key] = torch.sqrt(2 - 2 * torch.clamp(dist_mat[key], min=-1, max=1))
                            dist_mat_mean[key] = \
                                0.9 * ((dist_mat[key].sort(1)[0][:, 0] / dist_mat[key].sort(1)[0][:, 1]) < 0.8).sum() + \
                                0.1 * ((dist_mat[key].sort(1)[0][:, 0] / dist_mat[key].sort(1)[0][:, 1]) < 0.9).sum()

                            if dist_mat_mean[key] > now_dist:
                                # 记录 最 合适的键值
                                now_dist = dist_mat_mean[key]
                                suit_key = key

                        desc1 = desc1_dict[suit_key]
                        desc2 = desc2_dict[suit_key]
                    elif version == 5:
                        coord1, score1, desc1_dict, desc1_netvlad_dict = keypoint_net(image, ['perspective', 'rotate'],
                                                                                      is_need_netvlad=True)
                        coord2, score2, desc2_dict, desc2_netvlad_dict = keypoint_net(warped_image,
                                                                                      ['perspective', 'rotate'],
                                                                                      is_need_netvlad=True)
                        netvlad_dist_list = {}
                        suit_key = None
                        now_dist = 999
                        for key in desc1_dict.keys():
                            B, C, Hc, Wc = desc1_dict[key].shape
                            netvlad_dist_list[key] = torch.norm((desc1_netvlad_dict[key] - desc2_netvlad_dict[key]),
                                                                p=2, dim=1)

                            if netvlad_dist_list[key] < now_dist:
                                # 记录 最 合适的键值
                                now_dist = netvlad_dist_list[key]
                                suit_key = key

                        desc1 = desc1_dict[suit_key]
                        desc2 = desc2_dict[suit_key]
                    elif version == 6:
                        coord1, score1, desc1_dict, desc1_netvlad_dict = keypoint_net(image, ['perspective', 'rotate'],
                                                                                      is_need_netvlad=True)
                        coord2, score2, desc2_dict, desc2_netvlad_dict = keypoint_net(warped_image,
                                                                                      ['perspective', 'rotate'],
                                                                                      is_need_netvlad=True)
                        netvlad_dist_list = {}
                        suit_key = None
                        now_dist = 999
                        for key in desc1_dict.keys():
                            netvlad_dist_list[key] = torch.norm((desc1_netvlad_dict[key] - desc2_netvlad_dict[key]),
                                                                p=2, dim=1)

                        netvlad_dist = torch.tensor([netvlad_dist_list[key] for key in netvlad_dist_list])
                        netvlad_cos = 1 - 0.5 * netvlad_dist ** 2

                        netvlad_cos = F.softmax(netvlad_cos, dim=0)

                        tmp_idx = 0
                        for key in desc1_dict.keys():
                            desc1_dict[key] = desc1_dict[key] * netvlad_cos[tmp_idx]
                            desc2_dict[key] = desc2_dict[key] * netvlad_cos[tmp_idx]
                            tmp_idx = tmp_idx + 1

                        desc1 = torch.cat([desc1_dict[key] for key in desc1_dict.keys()], dim=1)
                        desc2 = torch.cat([desc2_dict[key] for key in desc2_dict.keys()], dim=1)
                    elif version == 7:
                        coord1, score1, desc1_dict, desc1_netvlad_dict = keypoint_net(image, [0, 1],
                                                                                      is_need_netvlad=True)
                        coord2, score2, desc2_dict, desc2_netvlad_dict = keypoint_net(warped_image, [0, 1],
                                                                                      is_need_netvlad=True)
                        netvlad_dist_list = {}
                        for key in desc1_dict.keys():
                            netvlad_dist_list[key] = torch.norm((desc1_netvlad_dict[key] - desc2_netvlad_dict[key]),
                                                                p=2, dim=1)

                        netvlad_dist = torch.tensor([netvlad_dist_list[key] for key in netvlad_dist_list])
                        netvlad_cos = 1 - 0.5 * netvlad_dist ** 2  # as score

                        if netvlad_cos.min() < 0.6:  # 出现一个评分特别小的
                            key_dict = ['perspective', 'rotate']
                            suit_key = netvlad_dist.argmin(dim=0)
                            desc1 = desc1_dict[key_dict[suit_key]]
                            desc2 = desc2_dict[key_dict[suit_key]]
                        else:
                            netvlad_cos_factor = F.softmax(netvlad_cos, dim=0)
                            tmp_idx = 0
                            for key in desc1_dict.keys():
                                desc1_dict[key] = desc1_dict[key] * netvlad_cos_factor[tmp_idx]
                                desc2_dict[key] = desc2_dict[key] * netvlad_cos_factor[tmp_idx]
                                tmp_idx = tmp_idx + 1
                            desc1 = torch.cat([desc1_dict[key] for key in desc1_dict.keys()], dim=1)
                            desc2 = torch.cat([desc2_dict[key] for key in desc2_dict.keys()], dim=1)
                    elif version == 8:
                        coord1, score1, desc1_dict, desc1_netvlad_dict = keypoint_net(image, [0, 1],
                                                                                      is_need_netvlad=True)
                        coord2, score2, desc2_dict, desc2_netvlad_dict = keypoint_net(warped_image, [0, 1],
                                                                                      is_need_netvlad=True)
                        netvlad_dist_list = {}
                        for key in desc1_dict.keys():
                            netvlad_dist_list[key] = torch.norm((desc1_netvlad_dict[key] - desc2_netvlad_dict[key]),
                                                                p=2, dim=1)

                        netvlad_dist = torch.tensor([netvlad_dist_list[key] for key in netvlad_dist_list])
                        netvlad_cos = 1 - 0.5 * netvlad_dist ** 2  # as score

                        if netvlad_cos[0] < 0.64 and netvlad_cos[1] < 0.59:  # 出现一个评分特别小的
                            key_dict = ['perspective', 'rotate']
                            suit_key = netvlad_dist.argmin(dim=0)
                            desc1 = desc1_dict[key_dict[suit_key]]
                            desc2 = desc2_dict[key_dict[suit_key]]
                        else:
                            netvlad_cos_factor = F.softmax(netvlad_cos, dim=0)
                            tmp_idx = 0
                            for key in desc1_dict.keys():
                                desc1_dict[key] = desc1_dict[key] * netvlad_cos_factor[tmp_idx]
                                desc2_dict[key] = desc2_dict[key] * netvlad_cos_factor[tmp_idx]
                                tmp_idx = tmp_idx + 1
                            desc1 = torch.cat([desc1_dict[key] for key in desc1_dict.keys()], dim=1)
                            desc2 = torch.cat([desc2_dict[key] for key in desc2_dict.keys()], dim=1)
                    elif version == 9:
                        coord1, score1, desc1_dict, desc1_netvlad_dict = keypoint_net(image,
                                                                                      ['perspective', 'rotate',
                                                                                       'scale'],
                                                                                      is_need_netvlad=True)
                        coord2, score2, desc2_dict, desc2_netvlad_dict = keypoint_net(warped_image,
                                                                                      ['perspective', 'rotate',
                                                                                       'scale'],
                                                                                      is_need_netvlad=True)
                        netvlad_dist_list = {}
                        suit_key = None
                        now_dist = 999
                        for key in desc1_dict.keys():
                            netvlad_dist_list[key] = torch.norm((desc1_netvlad_dict[key] - desc2_netvlad_dict[key]),
                                                                p=2, dim=1)

                        netvlad_dist = torch.tensor([netvlad_dist_list[key] for key in netvlad_dist_list])
                        netvlad_cos = 1 - 0.5 * netvlad_dist ** 2

                        netvlad_cos = F.softmax(netvlad_cos, dim=0)

                        tmp_idx = 0
                        for key in desc1_dict.keys():
                            desc1_dict[key] = desc1_dict[key] * netvlad_cos[tmp_idx]
                            desc2_dict[key] = desc2_dict[key] * netvlad_cos[tmp_idx]
                            tmp_idx = tmp_idx + 1

                        desc1 = torch.cat([desc1_dict[key] for key in desc1_dict.keys()], dim=1)
                        desc2 = torch.cat([desc2_dict[key] for key in desc2_dict.keys()], dim=1)
                    elif version == 10:
                        coord1, score1, desc1_dict, desc1_netvlad_dict = keypoint_net(image,
                                                                                      ['perspective', 'rotate',
                                                                                       'scale'],
                                                                                      is_need_netvlad=True)
                        coord2, score2, desc2_dict, desc2_netvlad_dict = keypoint_net(warped_image,
                                                                                      ['perspective', 'rotate',
                                                                                       'scale'],
                                                                                      is_need_netvlad=True)
                        netvlad_dist_list = {}
                        for key in desc1_dict.keys():
                            netvlad_dist_list[key] = torch.norm((desc1_netvlad_dict[key] - desc2_netvlad_dict[key]),
                                                                p=2, dim=1)

                        netvlad_dist = torch.tensor([netvlad_dist_list[key] for key in netvlad_dist_list])
                        netvlad_cos = 1 - 0.5 * netvlad_dist ** 2

                        netvlad_cos = F.softmax(netvlad_cos, dim=0)
                        netvlad_cos[netvlad_cos < 0.3333] = 0  # 嗯，挺合理的吧，把小于均值的踢了

                        tmp_idx = 0
                        for key in desc1_dict.keys():
                            desc1_dict[key] = desc1_dict[key] * netvlad_cos[tmp_idx]
                            desc2_dict[key] = desc2_dict[key] * netvlad_cos[tmp_idx]
                            tmp_idx = tmp_idx + 1

                        desc1 = torch.cat([desc1_dict[key] for key in desc1_dict.keys()], dim=1)
                        desc2 = torch.cat([desc2_dict[key] for key in desc2_dict.keys()], dim=1)

                    B, C, Hc, Wc = desc1.shape

                    # Scores & Descriptors
                    score1 = torch.cat([coord1, score1], dim=1).view(3, -1).t().cpu().numpy()
                    score2 = torch.cat([coord2, score2], dim=1).view(3, -1).t().cpu().numpy()
                    desc1 = desc1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
                    desc2 = desc2.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()

                    # Filter based on confidence threshold
                    desc1 = desc1[score1[:, 2] > conf_threshold, :]
                    desc2 = desc2[score2[:, 2] > conf_threshold, :]
                    score1 = score1[score1[:, 2] > conf_threshold, :]
                    score2 = score2[score2[:, 2] > conf_threshold, :]

                    # 映射一下，看着舒服点。。。就是重新给个名字，内存没变的。。。
                    desc = desc1
                    warped_desc = desc2
                    keypoints = score1
                    warped_keypoints = score2

                    # 保留k个分数最高的点，输出的keypoints为(2, -1)  desc为(C, -1)
                    keypoints, desc = select_k_best(keypoints, desc, params['top_k'])
                    warped_keypoints, warped_desc = select_k_best(warped_keypoints, warped_desc, params['top_k'])
                    # end_time = time.time()
                    # detect_time += end_time - start_time

                # 暴力匹配（其实就是最近邻的样子）
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # KNN do cross!
                matches = bf.match(desc, warped_desc)  # 获取配对
                matches = sorted(matches, key=lambda x: x.distance)  # 距离从小到大，排个序后面方便操作

                end_time = time.time()
                detect_time += end_time - start_time

                # 在配对成功的点对里筛选描述符空间距离合适的点对
                matches_dis = np.array([m.distance for m in matches])  # 获取每个匹配的距离，整合成numpy
                matches_idx = (matches_dis < params['des_dis'])  # True or False
                matches_query_idx = np.array([m.queryIdx for m in matches])[matches_idx]  # 获取配对中原图坐标对应下标
                matches_train_idx = np.array([m.trainIdx for m in matches])[matches_idx]  # 获取配对中目标图坐标对应下标
                m_keypoints = keypoints[matches_query_idx, :]  # 获取配对点坐标
                m_warped_keypoints = warped_keypoints[matches_train_idx, :]  # 获取配对点坐标

                # 分析描述符匹配准确率
                real_H = sample['homography'].squeeze().numpy()  # 获取变换矩阵
                m_d_warped_keypoints = warp_keypoints(m_warped_keypoints, np.linalg.inv(real_H))  # double warped hhh
                coord_dist = np.linalg.norm(m_d_warped_keypoints - m_keypoints, axis=-1)  # 坐标距离--欧氏距离
                correct_matches_nbr = int((coord_dist < 3).sum())  # 正确配对数量统计
                if i == 104:
                    correct_matches_nbr = int((coord_dist < 3).sum())  # 正确配对数量统计
                correct_matches_nbr_sum += correct_matches_nbr  # 合理正确配对数量
                matches_nbr = int((matches_dis < params['des_dis']).sum())
                matches_nbr_sum += matches_nbr

                # 获取可重复性
                true_warped_keypoints = warp_keypoints(keypoints, real_H)
                calc_warped_keypoints = warped_keypoints.copy()
                N1 = true_warped_keypoints.shape[0]
                N2 = warped_keypoints.shape[0]
                true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
                calc_warped_keypoints = np.expand_dims(calc_warped_keypoints, 0)
                norm = np.linalg.norm(true_warped_keypoints - calc_warped_keypoints, ord=None, axis=2)  # 求欧式距离
                if N2 != 0:
                    min1 = np.min(norm, axis=1)
                    correct1 = (min1 <= 3)  # 对true_warped_keypoints来讲，可能对应点，一个对一个
                    count1 = np.sum(correct1)  # 总数量
                if N1 != 0:
                    min2 = np.min(norm, axis=0)
                    correct2 = (min2 <= 3)  # 对warped_keypoints再来一次操作
                    count2 = np.sum(correct2)
                if N1 + N2 > 0:
                    rep = (count1 + count2) / (N1 + N2)
                else:
                    rep = 0

                # 分析单应性矩阵变换精度
                if m_keypoints.shape[0] < 4 or m_warped_keypoints.shape[0] < 4:
                    H_corner_dist = 999  # 连4个点都没有，根本找不到单应性矩阵，所以999，H大写是因为前面都是大写的，这里就不变了
                else:
                    calc_H, mask = cv2.findHomography(m_keypoints, m_warped_keypoints, cv2.RANSAC, 3, maxIters=5000)
                    if calc_H is None:
                        H_corner_dist = 999
                    _, _, H, W = sample['image'].shape  # 尺寸拉
                    # 单应性矩阵计算
                    corners = np.array([[0, 0, 1],
                                        [0, H - 1, 1],
                                        [W - 1, 0, 1],
                                        [W - 1, H - 1, 1]])
                    real_warped_corners = np.dot(corners, np.transpose(real_H))
                    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                    calc_warped_corners = np.dot(corners, np.transpose(calc_H))
                    calc_warped_corners = calc_warped_corners[:, :2] / calc_warped_corners[:, 2:]
                    H_corner_dist = np.mean(np.linalg.norm(real_warped_corners - calc_warped_corners, axis=1))  # 获取距离
                    correctness1.append(float(H_corner_dist <= 1))
                    correctness3.append(float(H_corner_dist <= 3))
                    correctness5.append(float(H_corner_dist <= 5))

                # 类型的转变 获取key点和扭曲点
                keypoints = \
                    [cv2.KeyPoint(keypoints[i][0], keypoints[i][1], 1) for i in range(keypoints.shape[0])]
                warped_keypoints = \
                    [cv2.KeyPoint(warped_keypoints[i][0], warped_keypoints[i][1], 1) for i in
                     range(warped_keypoints.shape[0])]

                # 画他，连线的那种，并且保存
                if params['draw_corr']:
                    # 选正确点
                    matches_new_idx = np.where(coord_dist < 3)
                    if i == 104:
                        matches_new_idx = np.where(coord_dist < 3)
                    matches_new = tuple(matches[tmp] for tmp in matches_new_idx[0])
                    img3 = cv2.drawMatches(img1, keypoints, img2, warped_keypoints,
                                           matches_new, None, flags=2)
                else:
                    img3 = cv2.drawMatches(img1, keypoints, img2, warped_keypoints,
                                           matches[0: matches_nbr], None, flags=2)

                # 保存部分
                cv2.imwrite(catch_point_dir + str(i) + '.png', cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))  # 图片
                cv2.imwrite(catch_point_dir + str(i) + 'sp.png', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  # 图片
                cv2.imwrite(catch_point_dir + str(i) + 'tp.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))  # 图片
                with open(catch_point_dir + '000nbr.txt', 'a') as f:  # 保存点数信息
                    write_str = '{:<4d} matches_nbr: {:<4d} correct_matches_nbr: {:<4d} ' \
                                'H_corner_dist: {:.3f}  rep: {:.3f}  '.format(
                        i,
                        matches_nbr,
                        correct_matches_nbr,
                        H_corner_dist,
                        rep)
                    if version == 4:
                        for key in dist_mat_mean.keys():
                            write_str = write_str + key + ': {:.3f}  '.format(dist_mat_mean[key])
                    if version == 5 or version == 6:
                        for tmp_idx in range(len(netvlad_cos)):
                            write_str = write_str + key + ': {:.3f}  '.format(netvlad_cos[tmp_idx])
                    write_str = write_str + '\n'

                    f.write(write_str)

            correctness1 = np.mean(correctness1)
            correctness3 = np.mean(correctness3)
            correctness5 = np.mean(correctness5)
            with open(catch_point_dir + '000nbr.txt', 'a') as f:  # 保存终极信息
                f.write('matches_nbr_sum: {:<8d} correct_matches_nbr_sum: {:<8d} correctness1: {:.3f} '
                        'correctness3: {:.3f} correctness5: {:.3f} detect_time: {:.4f}\n'.format(matches_nbr_sum,
                                                                                                 correct_matches_nbr_sum,
                                                                                                 correctness1,
                                                                                                 correctness3,
                                                                                                 correctness5,
                                                                                                 detect_time))
                # 低的原因是那边用的是共同的point
            print(detect_time)


if __name__ == '__main__':
    # file_path = curPath + '/step2_model.ckpt'
    file_path = '/home/featurize/6times_model.ckpt'
    main(file_path)
