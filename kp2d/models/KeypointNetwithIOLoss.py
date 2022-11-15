# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import get_cmap

from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.networks.keypoint_preact_resnet import KeypointPreactResnet
from kp2d.utils.image import image_grid, to_color_normalized, to_gray_normalized
from kp2d.utils.keypoints import draw_keypoints, get_border_mask, get_uv_unnorm, get_uv_norm, warp_homography_batch


def get_inline_netvlad_loss(source_one_feats, target_one_feats):
    """
    输入的是 B * C 的 tensor，不是tuple哦
    feat 都是L2归一化了的
    Parameters
    ----------
    source_one_feats
    target_one_feats

    Returns
    -------

    """
    B, C = source_one_feats.shape
    eyes = 10 * torch.eye(B, B).to(source_one_feats.device).detach()
    # 获取欧式距离矩阵，源 --> 目标
    d_mat = torch.mm(source_one_feats, target_one_feats.t())  # 变成一个B * B矩阵，对角线处是正确的即pos_var
    d_mat = torch.sqrt(2 - 2 * torch.clamp(d_mat, min=-1, max=1))  # 转化成欧式距离
    recall = (torch.diag(d_mat) == torch.min(d_mat, dim=1)[0]).float().mean()  # 准确预测

    d_mat = d_mat + eyes  # 对角线是pos不要来影响我选择neg
    loss = 0

    # 筛选距离最近的
    neg_idx = torch.argmin(d_mat, dim=1)  # 明明不是一起的，距离还这么近，哼

    anchor_var = source_one_feats  # B * C
    pos_var = target_one_feats
    neg_var = target_one_feats[neg_idx, :]  # 和上面对应的，和最难样本保持距离
    loss = loss + torch.nn.functional.triplet_margin_loss(anchor_var, pos_var, neg_var, margin=1)

    # 筛选距离最近的
    neg_idx = torch.argmin(d_mat, dim=0)  # 明明不是一起的，距离还这么近，哼

    anchor_var = target_one_feats  # B * C
    pos_var = source_one_feats
    neg_var = source_one_feats[neg_idx, :]  # 和上面对应的，和最难样本保持距离

    loss = loss + torch.nn.functional.triplet_margin_loss(anchor_var, pos_var, neg_var, margin=1)

    return 0.5 * loss, recall


def get_dist_mat(desc1, desc2):
    dist_mat = torch.mm(desc1.t(), desc2)
    dist_mat = torch.sqrt(2 - 2 * dist_mat.clamp(min=-1, max=1))
    return dist_mat


def get_hard_triplet_half(src_desc, tgt_desc, points_raw, relax_field, is_eval=False):
    """
    不选取最小，即 get_hard_triplet 只执行一半
    """

    dist_mat = get_dist_mat(src_desc, tgt_desc)
    dist_mat_sorted, idx = torch.sort(dist_mat, dim=1)
    candidates = idx.t()

    match_k_x = points_raw[0, candidates]  # 一个true对应k个匹配点
    match_k_y = points_raw[1, candidates]

    true_x = points_raw[0]
    true_y = points_raw[1]

    # Compute recall as the number of correct matches, i.e. the first match is the correct one
    correct_matches = (abs(match_k_x[0] - true_x) == 0) & (abs(match_k_y[0] - true_y) == 0)
    recall = correct_matches.float().sum() / src_desc.size(1)

    if is_eval:
        return None, recall

    # Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field)
    correct_idx = (abs(match_k_x - true_x) <= relax_field) & (abs(match_k_y - true_y) <= relax_field)

    # Get hardest negative example as an incorrect match and with the smallest descriptor distance
    incorrect_first = dist_mat_sorted.t()
    incorrect_first[correct_idx] = 2.0
    dist_a2n, incorrect_first = incorrect_first.min(dim=0)
    incorrect_first_index = candidates.gather(0, incorrect_first.unsqueeze(0)).squeeze()

    anchor_var = src_desc
    pos_var = tgt_desc
    neg_var = tgt_desc[:, incorrect_first_index]

    return (anchor_var, pos_var, neg_var, dist_a2n), recall  # 顺便返回一个 anchor 和 neg 的距离，方便后期筛选


def get_hard_triplet(src_desc, tgt_desc, points_raw, relax_field, eval_only=False):
    (anchor_var_st, pos_var_st, neg_var_st, dist_a2n_st), recall_st = get_hard_triplet_half(src_desc, tgt_desc,
                                                                                            points_raw, relax_field,
                                                                                            eval_only)
    (anchor_var_ts, pos_var_ts, neg_var_ts, dist_a2n_ts), recall_ts = get_hard_triplet_half(tgt_desc, src_desc,
                                                                                            points_raw, relax_field,
                                                                                            eval_only)

    dist_a2n_combine = torch.cat([dist_a2n_st.unsqueeze(0), dist_a2n_ts.unsqueeze(0)], dim=0)
    anchor_var_combine = torch.cat([anchor_var_st.unsqueeze(0), anchor_var_ts.unsqueeze(0)], dim=0)
    pos_var_combine = torch.cat([pos_var_st.unsqueeze(0), pos_var_ts.unsqueeze(0)], dim=0)
    neg_var_combine = torch.cat([neg_var_st.unsqueeze(0), neg_var_ts.unsqueeze(0)], dim=0)

    nearest_idx = dist_a2n_combine.argmin(0).repeat(64, 1).unsqueeze(0)
    anchor_var = anchor_var_combine.gather(0, nearest_idx).squeeze()
    pos_var = pos_var_combine.gather(0, nearest_idx).squeeze()
    neg_var = neg_var_combine.gather(0, nearest_idx).squeeze()
    recall = 0.5 * (recall_st + recall_ts)

    return (anchor_var, pos_var, neg_var), recall


def build_descriptor_loss(source_des: torch.Tensor,
                          target_des: torch.Tensor,
                          source_points_norm: torch.Tensor,
                          target_points_norm: torch.Tensor,
                          target_points_unnorm: torch.Tensor,
                          keypoint_mask=None,
                          relax_field=8,
                          margins=0.2,
                          eval_only=False,
                          do_cross=False):
    """
    如何该函数中存在含batch的和不含batch的，我会在不含batch的变量前面加a_
    Parameters
    ----------
    do_cross: bool
        要不是互选最小值
    margins: float
        triple loss margins
    keypoint_mask: None or torch.Tensor
        选取合适的点
    relax_field: int
        小于field的点被认为正确
    source_des: torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des: torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points_norm: torch.Tensor (B,H/8,W/8,2)
        Source image key points
    target_points_norm: torch.Tensor (B,H/8,W/8,2)
        Target image key points
    target_points_unnorm: torch.Tensor (B,2,H/8,W/8)
        Target image keypoints un normalized
    eval_only: bool
        Computes only recall without the loss.
    Returns
    -------
    loss: torch.Tensor
        Descriptor loss.
    recall: torch.Tensor
        Descriptor match recall.
    """
    device = source_des.device
    batch_size, C, _, _ = source_des.shape
    loss, recall = 0., 0.

    # -------------------- divide them one by one, so value behind have no batch dimension --------------------
    for cur_idx in range(batch_size):
        # 一个个来能省很多显存啊，但是会慢大概 1 image/s
        a_src_desc = torch.nn.functional.grid_sample(source_des[cur_idx].unsqueeze(0),
                                                     source_points_norm[cur_idx].unsqueeze(0),
                                                     align_corners=False).squeeze()
        a_tgt_desc = torch.nn.functional.grid_sample(target_des[cur_idx].unsqueeze(0),
                                                     target_points_norm[cur_idx].unsqueeze(0),
                                                     align_corners=False).squeeze()
        a_src_desc = F.normalize(a_src_desc, p=2, dim=0)
        a_tgt_desc = F.normalize(a_tgt_desc, p=2, dim=0)

        if keypoint_mask is None:
            a_src_desc = a_src_desc.view(C, -1)
            a_tgt_desc = a_tgt_desc.view(C, -1)
            a_target_points_raw = target_points_unnorm[cur_idx].view(2, -1)
        else:
            a_keypoint_mask = keypoint_mask[cur_idx].squeeze()
            n_feat = a_keypoint_mask.sum().item()  # 点数太少直接返回
            if n_feat < 20:
                continue
            a_src_desc = a_src_desc[:, a_keypoint_mask]
            a_tgt_desc = a_tgt_desc[:, a_keypoint_mask]
            a_target_points_raw = target_points_unnorm[cur_idx][:, a_keypoint_mask]

        if do_cross is False:
            (anchor_var, pos_var, neg_var, _), tmp_recall = get_hard_triplet_half(a_src_desc, a_tgt_desc,
                                                                                  a_target_points_raw,
                                                                                  relax_field, eval_only)
        else:
            (anchor_var, pos_var, neg_var, _), tmp_recall = get_hard_triplet(a_src_desc, a_tgt_desc,
                                                                             a_target_points_raw,
                                                                             relax_field, eval_only)

        recall += float(1.0 / batch_size) * tmp_recall
        loss += float(1.0 / batch_size) * torch.nn.functional.triplet_margin_loss(anchor_var.t(), pos_var.t(),
                                                                                  neg_var.t(), margin=margins)

    return loss, recall


class KeypointNetwithIOLoss(torch.nn.Module):
    """
    Model class encapsulating the KeypointNet and the IONet.

    Parameters
    ----------
    keypoint_loss_weight: float
        Keypoint loss weight.
    descriptor_loss_weight: float
        Descriptor loss weight.
    score_loss_weight: float
        Score loss weight.
    keypoint_net_learning_rate: float
        Keypoint net learning rate.
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample dense descriptor map.
    do_cross: bool
        Predict keypoints outside cell borders.
    with_drop : bool
        Use dropout.
    descriptor_loss: bool
        Use descriptor loss.
    kwargs : dict
        Extra parameters
    """

    def __init__(self, keypoint_loss_weight=1.0, descriptor_loss_weight=2.0, score_loss_weight=1.0,
                 keypoint_net_learning_rate=0.001, use_color=True, do_upsample=True, do_cross=True,
                 descriptor_loss=True, with_drop=True, use_new_descriptor_loss=True, add_score_addition_loss=True,
                 keypoint_net_type='KeypointNet', add_desc_relate_score_loss=True, train_which_desc='', **kwargs):

        super().__init__()

        self.keypoint_loss_weight = keypoint_loss_weight
        self.descriptor_loss_weight = descriptor_loss_weight
        self.score_loss_weight = score_loss_weight
        self.keypoint_net_learning_rate = keypoint_net_learning_rate
        self.optim_params = []

        self.top_k2 = 300
        self.relax_field = 4  # 给desc 用的
        self.dist_threshold = 4  # 给dist 用的

        self.use_color = use_color
        self.descriptor_loss = descriptor_loss
        self.use_new_descriptor_loss = use_new_descriptor_loss
        self.add_score_addition_loss = add_score_addition_loss
        self.add_desc_relate_score_loss = add_desc_relate_score_loss
        self.train_which_desc = train_which_desc

        self.vis = {}

        # Initialize KeypointNet
        if keypoint_net_type == 'KeypointNet':
            self.keypoint_net = KeypointNet(use_color=use_color, do_upsample=do_upsample, with_drop=with_drop,
                                            do_cross=do_cross)
        elif keypoint_net_type == 'KeypointResnet':
            self.keypoint_net = KeypointResnet(with_drop=with_drop)
        elif keypoint_net_type == 'KeypointPreactResnet':
            self.keypoint_net = KeypointPreactResnet(with_drop=with_drop)
        else:
            raise NotImplemented('Keypoint net type not supported {}'.format(keypoint_net_type))

        self.keypoint_net = self.keypoint_net.cuda()
        self.add_optimizer_params('KeypointNet', self.keypoint_net.parameters(), keypoint_net_learning_rate)

        if torch.cuda.current_device() == 0:
            print('KeypointNetwithIOLoss::with descriptor loss {}'.format(self.descriptor_loss))

    def add_optimizer_params(self, name, params, lr):
        self.optim_params.append(
            {'name': name, 'lr': lr, 'original_lr': lr,
             'params': filter(lambda p: p.requires_grad, params)})

    def train_basic(self, data, debug=False):
        loss_2d_total = {}
        recall_2d_total = {}
        loss_2d = 0
        recall_2d = 0
        if self.training:

            # -------------------- get some info --------------------
            B, _, H, W = data['img'].shape
            device = data['img'].device

            input_img = data['img']
            input_img_aug = data['img_aug']
            homography = data['homography']

            input_img = to_color_normalized(input_img)
            input_img_aug = to_color_normalized(input_img_aug)

            # -------------------- get network outputs --------------------
            if self.add_desc_relate_score_loss:
                source_score, source_uv_pred, source_desc, src_grid_desc = self.keypoint_net(input_img_aug,
                                                                                             need_grid_desc=True)
                target_score, target_uv_pred, target_desc, tgt_grid_desc = self.keypoint_net(input_img,
                                                                                             need_grid_desc=True)
            else:
                source_score, source_uv_pred, source_desc = self.keypoint_net(input_img_aug)
                target_score, target_uv_pred, target_desc = self.keypoint_net(input_img)

            _, _, Hc, Wc = target_score.shape

            # -------------------- get all kinds of coordinate --------------------
            target_uv_norm = get_uv_norm(target_uv_pred, H, W)
            source_uv_norm = get_uv_norm(source_uv_pred, H, W)
            source_uv_warped_norm = warp_homography_batch(source_uv_norm, homography)
            source_uv_warped = get_uv_unnorm(source_uv_warped_norm, H, W)
            target_uv_warped_norm = warp_homography_batch(target_uv_norm, torch.inverse(homography))
            target_uv_warped = get_uv_unnorm(target_uv_warped_norm, H, W)

            # -------------------- get border mask --------------------
            border_mask_ori = get_border_mask(B, Hc, Wc, device)
            border_mask_ori = border_mask_ori.gt(1e-3).to(device)  # return True or False
            # out-of-border(oob) mask source-to-target(s2t)
            oob_mask_s2t = source_uv_warped_norm[:, :, :, 0].lt(1) & source_uv_warped_norm[:, :, :, 0].gt(
                -1) & source_uv_warped_norm[:, :, :, 1].lt(1) & source_uv_warped_norm[:, :, :, 1].gt(-1)
            border_mask_s = border_mask_ori & oob_mask_s2t  # used in source img network output
            # out-of-border(oob) mask target-to-source(t2s)
            oob_mask_t2s = target_uv_warped_norm[:, :, :, 0].lt(1) & target_uv_warped_norm[:, :, :, 0].gt(
                -1) & target_uv_warped_norm[:, :, :, 1].lt(1) & target_uv_warped_norm[:, :, :, 1].gt(-1)
            border_mask_t = border_mask_ori & oob_mask_t2s

            # -------------------- get dist mat between source and target, get loc loss --------------------
            d_uv_mat_abs = torch.abs(
                source_uv_warped.view(B, 2, -1).unsqueeze(3) - target_uv_pred.view(B, 2, -1).unsqueeze(2))
            d_uv_l2_mat = torch.norm(d_uv_mat_abs, p=2, dim=1)
            d_uv_l2_min, d_uv_l2_min_index = d_uv_l2_mat.min(dim=2)

            dist_norm_valid_mask = d_uv_l2_min.lt(self.dist_threshold) & border_mask_s.view(B, Hc * Wc)

            if self.add_desc_relate_score_loss:
                cos_desc_mat = torch.bmm(src_grid_desc.view(B, 64, -1).permute(0, 2, 1), tgt_grid_desc.view(B, 64, -1))
                cos_desc = cos_desc_mat.gather(2, d_uv_l2_min_index.unsqueeze(2)).squeeze()[dist_norm_valid_mask]
                cos_desc = 0.5 * (cos_desc + 1)  # 变换到0~1之间
                cos_desc_mat_max, _ = cos_desc_mat.max(dim=2)
                cos_desc_mat_max = (cos_desc_mat_max[dist_norm_valid_mask] + 1) * 0.5

            loc_loss = d_uv_l2_min[dist_norm_valid_mask].mean()
            loss_2d_total['loc_loss'] = self.keypoint_loss_weight * loc_loss

            # -------------------- get score loss(usp and mse) --------------------
            target_score_associated = \
                target_score.view(B, Hc * Wc).gather(1, d_uv_l2_min_index).view(B, Hc, Wc).unsqueeze(1)
            dist_norm_valid_mask = dist_norm_valid_mask.view(B, Hc, Wc).unsqueeze(1) & border_mask_s.unsqueeze(1)
            d_uv_l2_min = d_uv_l2_min.view(B, Hc, Wc).unsqueeze(1)
            loc_err = d_uv_l2_min[dist_norm_valid_mask]
            target_score_associated = (target_score_associated - target_score_associated.min().detach()) / \
                                      (target_score_associated.max() - target_score_associated.min()).detach()
            source_score = (source_score - source_score.min().detach()) / \
                           (source_score.max() - source_score.min()).detach()
            loc_err_new = loc_err - loc_err.mean()
            usp_loss = ((target_score_associated[dist_norm_valid_mask] +
                         source_score[dist_norm_valid_mask]) * loc_err_new).mean()

            if self.add_desc_relate_score_loss:
                # 强制(cos_desc_mat_max - cos_desc)趋于0，并且分数根据前者进行变动
                hehe = ((cos_desc_mat_max - cos_desc) * (target_score_associated[dist_norm_valid_mask] +
                                                         source_score[dist_norm_valid_mask])).mean()
                usp_loss = usp_loss + hehe

            # 插值其实影响也不是很大
            target_score_resampled = torch.nn.functional.grid_sample(target_score, source_uv_warped_norm.detach(),
                                                                     mode='bilinear', align_corners=True)
            score_loss = usp_loss.mean() + 2 * torch.nn.functional.mse_loss(
                target_score_resampled[border_mask_s.unsqueeze(1)],
                source_score[border_mask_s.unsqueeze(1)]).mean()
            loss_2d_total['score_loss'] = self.score_loss_weight * score_loss

            # -------------------- get score addition loss --------------------
            if self.add_score_addition_loss:
                score_var = (target_score_associated[dist_norm_valid_mask] + source_score[dist_norm_valid_mask]).var()
                score_addition_loss = F.relu(torch.exp(-score_var).view([]) - 0.77880)
                loss_2d_total['score_addition_loss'] = self.score_loss_weight * score_addition_loss  # scale like score
            else:
                loss_2d_total['score_addition_loss'] = 0

            # -------------------- get score descriptor loss ------------------
            if self.descriptor_loss:
                metric_loss0, recall_2d0 = build_descriptor_loss(source_desc, target_desc, source_uv_norm.detach(),
                                                                 source_uv_warped_norm.detach(), source_uv_warped,
                                                                 keypoint_mask=border_mask_s,
                                                                 relax_field=self.relax_field)
                if self.use_new_descriptor_loss:
                    metric_loss1, recall_2d1 = build_descriptor_loss(target_desc, source_desc,
                                                                     source_uv_warped_norm.detach(),
                                                                     source_uv_norm.detach(), source_uv_warped,
                                                                     keypoint_mask=border_mask_s,
                                                                     relax_field=self.relax_field)

                    metric_loss2, recall_2d2 = build_descriptor_loss(source_desc, source_desc, source_uv_norm.detach(),
                                                                     source_uv_norm.detach(), source_uv_pred,
                                                                     keypoint_mask=None,
                                                                     relax_field=self.relax_field,
                                                                     margins=0.5)

                    metric_loss3, recall_2d3 = build_descriptor_loss(target_desc, target_desc, target_uv_norm.detach(),
                                                                     target_uv_norm.detach(), target_uv_pred,
                                                                     keypoint_mask=None,
                                                                     relax_field=self.relax_field,
                                                                     margins=0.5)

                    metric_loss4, recall_2d4 = build_descriptor_loss(target_desc, source_desc, target_uv_norm.detach(),
                                                                     target_uv_warped_norm.detach(), target_uv_warped,
                                                                     keypoint_mask=border_mask_t,
                                                                     relax_field=self.relax_field)

                    metric_loss5, recall_2d5 = build_descriptor_loss(source_desc, target_desc,
                                                                     target_uv_warped_norm.detach(),
                                                                     target_uv_norm.detach(), target_uv_warped,
                                                                     keypoint_mask=border_mask_t,
                                                                     relax_field=self.relax_field)

                    metric_loss = 0.25 * (
                            metric_loss0 + metric_loss1 + metric_loss2 + metric_loss3 + metric_loss4 + metric_loss5)
                    recall_2d = 0.25 * (recall_2d0 + recall_2d1 + recall_2d4 + recall_2d5)
                else:
                    metric_loss = metric_loss0
                    recall_2d = recall_2d0

                loss_2d_total['desc_loss'] = self.descriptor_loss_weight * metric_loss * 2
            else:
                _, recall_2d = build_descriptor_loss(source_desc, target_desc, source_uv_norm, source_uv_warped_norm,
                                                     source_uv_warped, keypoint_mask=border_mask_s,
                                                     relax_field=self.relax_field, eval_only=True)

            # -------------------- get vis pictures --------------------
            if debug and torch.cuda.current_device() == 0:
                # Generate visualization data
                # 和训练无关
                # 目标图，或者你可以看作原来的图，因为是反的--------------------------------------
                vis_ori0 = (input_img[0].permute(1, 2, 0).detach().cpu().clone().squeeze())
                vis_ori0 -= vis_ori0.min()
                vis_ori0 /= vis_ori0.max()
                vis_ori0 = (vis_ori0 * 255).numpy().astype(np.uint8)

                if self.use_color is False:
                    vis_ori0 = cv2.cvtColor(vis_ori0, cv2.COLOR_GRAY2BGR)

                # 变换图，但是在这里算是源图---------------------------------------------------
                vis_ori1 = (input_img_aug[0].permute(1, 2, 0).detach().cpu().clone().squeeze())
                vis_ori1 -= vis_ori1.min()
                vis_ori1 /= vis_ori1.max()
                vis_ori1 = (vis_ori1 * 255).numpy().astype(np.uint8)

                if self.use_color is False:
                    vis_ori1 = cv2.cvtColor(vis_ori1, cv2.COLOR_GRAY2BGR)

                # 在目标图画点---------------------------------------------------------------
                _, top_k = target_score.view(B, -1).topk(self.top_k2, dim=1)  # JT: Target frame keypoints
                vis_ori0 = draw_keypoints(vis_ori0, target_uv_pred.view(B, 2, -1)[:, :, top_k[0].squeeze()],
                                          (0, 0, 255))

                _, top_k = source_score.view(B, -1).topk(self.top_k2, dim=1)  # JT: Warped Source frame keypoints
                vis_ori0 = draw_keypoints(vis_ori0, source_uv_warped.view(B, 2, -1)[:, :, top_k[0].squeeze()],
                                          (255, 0, 255))

                # 在源图画点---------------------------------------------------------------
                target_score_cd = target_score.clone().detach()
                target_score_cd[~oob_mask_t2s.unsqueeze(1)] = -1  # 直接把分数pass掉不就完事了，哎~
                _, top_k = target_score_cd.view(B, -1).topk(self.top_k2, dim=1)  # JT: Target frame keypoints
                vis_ori1 = draw_keypoints(vis_ori1, target_uv_warped.view(B, 2, -1)[:, :, top_k[0].squeeze()],
                                          (0, 0, 255))

                _, top_k = source_score.view(B, -1).topk(self.top_k2, dim=1)  # JT: Warped Source frame keypoints
                vis_ori1 = draw_keypoints(vis_ori1, source_uv_pred.view(B, 2, -1)[:, :, top_k[0].squeeze()],
                                          (255, 0, 255))

                # 画热力图-----------------------------------------------------------------
                cm = get_cmap('plasma')
                heatmap = target_score[0].detach().cpu().clone().numpy().squeeze()
                heatmap -= heatmap.min()
                heatmap /= heatmap.max()
                heatmap = cv2.resize(heatmap, (W, H))
                heatmap = cm(heatmap)[:, :, :3]

                # 改成适合tensorboardX的----------------------------------------------------
                self.vis['img_ori0'] = np.clip(vis_ori0, 0, 255).transpose((2, 0, 1))
                self.vis['img_ori1'] = np.clip(vis_ori1, 0, 255).transpose((2, 0, 1))
                self.vis['heatmap'] = np.clip(heatmap * 255, 0, 255).transpose((2, 0, 1))

            # 获取总的loss
            loss_2d = 0
            for key in loss_2d_total.keys():
                loss_2d = loss_2d + loss_2d_total[key]

        return loss_2d, recall_2d, loss_2d_total, recall_2d_total

    def train_netvlad(self, data):
        loss_2d_total = {}
        recall_2d_total = {}
        loss_2d = 0
        recall_2d = 0
        if self.training:
            # -------------------- get data --------------------
            B, _, H, W = data['img'].shape
            device = data['img'].device

            input_img = data['img']
            input_img_aug = data['img_aug']
            homography = data['homography']

            input_img = to_color_normalized(input_img)
            input_img_aug = to_color_normalized(input_img_aug)

            # -------------------- get networks output --------------------
            _, _, _, tgt_netvlad_desc = self.keypoint_net(input_img, need_grid_desc=False, is_need_netvlad=True)
            _, _, _, src_netvlad_desc = self.keypoint_net(input_img_aug, need_grid_desc=False, is_need_netvlad=True)

            # -------------------- get inline loss --------------------
            loss_2d_total['inline'], recall_2d_total['inline'] = get_inline_netvlad_loss(src_netvlad_desc,
                                                                                         tgt_netvlad_desc)

            # 获取总的loss
            for key in loss_2d_total.keys():
                loss_2d = loss_2d + loss_2d_total[key]
            for key in recall_2d_total.keys():
                recall_2d = recall_2d + recall_2d_total[key]
            recall_2d = recall_2d / len(recall_2d_total)

        return loss_2d, recall_2d, loss_2d_total, recall_2d_total

    def forward(self, data, debug=False):
        if self.train_which_desc == 'netvlad':
            loss_2d, recall_2d, loss_2d_total, recall_2d_total = self.train_netvlad(data)
        else:
            loss_2d, recall_2d, loss_2d_total, recall_2d_total = self.train_basic(data, debug)
        return loss_2d, recall_2d, loss_2d_total, recall_2d_total
