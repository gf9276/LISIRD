# Copyright 2020 Toyota Research Institute.  All rights reserved.
from math import pi

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F

from kp2d.datasets.augmentations import get_fixed_h
from kp2d.evaluation.descriptor_evaluation import (compute_homography,
                                                   compute_matching_score)
from kp2d.evaluation.detector_evaluation import compute_repeatability
from kp2d.utils.image import to_color_normalized, to_gray_normalized
import cv2
import operator
from kp2d.evaluation.descriptor_evaluation import select_k_best
from kp2d.evaluation.descriptor_evaluation import bf_match
from kp2d.utils.keypoints import get_uv_norm, warp_homography_batch, get_uv_unnorm, get_border_mask


def evaluate_keypoint_net(data_loader, keypoint_net, output_shape=(320, 240), top_k=300, use_color=True, version=1):
    """Keypoint net evaluation script.

    Parameters
    ----------
    version
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore, correct_matches_nbr, matches_nbr = [], [], [], [], [], []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
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

                image = to_color_normalized(sample['image'].cuda())
                warped_image = to_color_normalized(sample['warped_image'].cuda())
            else:
                image = to_gray_normalized(sample['image'].cuda())
                warped_image = to_gray_normalized(sample['warped_image'].cuda())

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

                    nbr1, _ = score1[0].view(-1).topk(int(top_k * factor[0]))
                    nbr1 = nbr1[-1]
                    score1[0][score1[0] >= nbr1] = 1
                    score1[0][score1[0] < nbr1] = 0

                    nbr2, _ = score2[0].view(-1).topk(int(top_k * factor[0]))
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

            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape': output_shape,
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score1,
                    'warped_prob': score2,
                    'desc': desc1,
                    'warped_desc': desc2}

            # Compute repeatabilty and localization error
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            # Compute matching score
            mscore, cmn, mn = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)
            correct_matches_nbr.append(cmn)
            matches_nbr.append(mn)

    return np.mean(repeatability), np.mean(localization_err), np.mean(correctness1), np.mean(correctness3), \
           np.mean(correctness5), np.mean(MScore), np.mean(correct_matches_nbr), np.mean(matches_nbr), \
           repeatability, localization_err, correctness1, correctness3, \
           correctness5, MScore, correct_matches_nbr, matches_nbr
