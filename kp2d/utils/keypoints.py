# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch


def get_uv_norm(uv_unnorm, H, W):
    uv_norm = uv_unnorm.clone()
    uv_norm[:, 0] = (uv_norm[:, 0] / (float(W - 1) / 2.)) - 1.
    uv_norm[:, 1] = (uv_norm[:, 1] / (float(H - 1) / 2.)) - 1.
    uv_norm = uv_norm.permute(0, 2, 3, 1)
    return uv_norm


def get_uv_unnorm(uv_norm, H, W):
    uv_unnorm = uv_norm.clone()
    uv_unnorm[:, :, :, 0] = (uv_unnorm[:, :, :, 0] + 1) * (float(W - 1) / 2.)
    uv_unnorm[:, :, :, 1] = (uv_unnorm[:, :, :, 1] + 1) * (float(H - 1) / 2.)
    uv_unnorm = uv_unnorm.permute(0, 3, 1, 2)
    return uv_unnorm


def get_border_mask(B, Hc, Wc, device):
    """
    周围清空一圈
    border 边缘嘛
    """
    board_mask = torch.ones(B, Hc, Wc, device=device)
    board_mask[:, 0] = 0
    board_mask[:, Hc - 1] = 0
    board_mask[:, :, 0] = 0
    board_mask[:, :, Wc - 1] = 0
    return board_mask


def warp_keypoints(keypoints, H):
    """Warp keypoints given a homography

    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.

    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def warp_homography_batch(sources, homographies):
    """Batch warp keypoints given homographies.

    Parameters
    ----------
    sources: torch.Tensor (B,H,W,C)
        Keypoints vector.
    homographies: torch.Tensor (B,3,3)
        Homographies.

    Returns
    -------
    warped_sources: torch.Tensor (B,H,W,C)
        Warped keypoints vector.
    """
    B, H, W, _ = sources.shape
    warped_sources = []
    for b in range(B):
        source = sources[b].clone()
        source = source.view(-1, 2)
        source = torch.addmm(homographies[b, :, 2], source, homographies[b, :, :2].t())
        source.mul_(1 / source[:, 2].unsqueeze(1))
        source = source[:, :2].contiguous().view(H, W, 2)
        warped_sources.append(source)
    return torch.stack(warped_sources, dim=0)


def draw_keypoints(img_l, top_uvz, color=(255, 0, 0), idx=0):
    """Draw keypoints on an image"""
    vis_xyd = top_uvz.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()
    vis = img_l.copy()
    cnt = 0
    for pt in vis_xyd[:, :2].astype(np.int32):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 2, color, -1)
    return vis
