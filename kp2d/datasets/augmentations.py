# Copyright 2020 Toyota Research Institute.  All rights reserved.
import copy
import random
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from math import pi
from PIL import Image
from kp2d.utils.image import (image_grid, to_color_normalized, to_gray_normalized)
from skimage import feature, exposure
from kp2d.utils.keypoints import get_uv_norm

# 去除黑边的操作
crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w, :]  # 定义裁切函数，后续裁切黑边使用


def rotate_image(img, angle, crop):
    """
    稍微修改过，完美
    angle: 旋转的角度
    crop: 是否需要进行裁剪，布尔向量
    """
    h, w = img.shape[:2]
    # 旋转角度的周期是360°
    angle %= 360
    # 计算仿射变换矩阵
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    # 如果需要去除黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2 + 0.5)
        y0 = int((h - h_crop) / 2 + 0.5)
        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def filter_dict(dict, keywords):
    """
    Returns only the keywords that are part of a dictionary

    Parameters
    ----------
    dict : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered

    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dict]


def resize_sample(sample, image_shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, which contains an input image.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    image_shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # image
    image_transform = transforms.Resize(image_shape, interpolation=image_interpolation)
    sample['image'] = image_transform(sample['image'])
    return sample


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    sample['image'] = transform(sample['image']).type(tensor_type)
    return sample


def cv22pil(img):
    # 带变色的，别乱玩
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image


def pil2cv2(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img


def change_wh(img):
    # 我希望宽长一点
    h, w = img.shape[:2]
    if h > w:
        img = np.transpose(img, (1, 0, 2))
    return img


def spatial_augment_sample(sample):
    """ Apply spatial augmentation to an image (flipping and random affine transformation)."""
    augment_image = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))  # 这个不行，哼
    ])
    sample['image'] = augment_image(sample['image'])
    sample['image'] = pil2cv2(sample['image'])  # 变成cv2
    sample['image'] = change_wh(sample['image'])
    # sample['image'] = rotate_image(sample['image'], random.randint(-15, 15), True)  # 我认为旋转裁剪本身就带缩放了，不管了
    sample['image'] = cv22pil(sample['image'])  # 变回PIL。。。
    return sample


def unnormalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """ Counterpart method of torchvision.transforms.Normalize."""
    for t, m, s in zip(tensor, mean, std):
        t.div_(1 / s).sub_(-m)
    return tensor


def get_rotate_pts(hw_ratio, pts, min_angle=None, max_angle=pi / 4, n_angles=100, angle_type=0):
    if angle_type == 0:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        if min_angle is not None:
            if pi / 3 > min_angle >= pi / 6:
                angles_add = np.linspace(min_angle, pi / 3, n_angles // 4)  # 旋转角度太大也不好
                angles = np.concatenate([-angles_add, angles_add, angles], axis=0)
    elif angle_type == 1:
        # 获取一系列旋转角度，选取 angle_type == 1 的时候，最好把旋转放在前面，防止出现0
        angles1 = np.linspace(-max_angle, -min_angle, n_angles // 2)
        angles2 = np.linspace(min_angle, max_angle, n_angles // 2)
        angles = np.concatenate([angles1, angles2], axis=0)
        angles = np.delete(angles, np.where(angles == 0.))

    angles = np.concatenate([[0.], angles], axis=0)
    n_angles = angles.size

    # 获取中心并获取n个旋转矩阵
    center = np.mean(pts, axis=0, keepdims=True)
    rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                   np.cos(angles)], axis=1), [-1, 2, 2])

    # 获取旋转后的点 np里的 tile 和 torch里的 repeat 是一样的样子
    rotated = np.matmul(np.tile(np.expand_dims(pts - center, axis=0), [n_angles, 1, 1]), rot_mat) + center

    # 筛选在范围内的点
    valid = np.where(np.all((rotated >= [-1., -hw_ratio]) & (rotated <= [1., hw_ratio]), axis=(1, 2)))[0]
    idx = valid[np.random.randint(valid.shape[0])]  # 聚在中间变化不大的会是多数啊
    pts = rotated[idx]

    return pts


def get_scale_pts(hw_ratio, pts, scaling_amplitude, n_scales=100):
    # 获取随机缩放系数，基于patch ratio过的图
    random_scales = np.random.normal(1, scaling_amplitude / 2, (n_scales))
    random_scales = np.clip(random_scales, 1 - scaling_amplitude / 2, 1 + scaling_amplitude / 2)

    scales = np.concatenate([[1.], random_scales], 0)
    center = np.mean(pts, axis=0, keepdims=True)  # 获取中心
    scaled = np.expand_dims(pts - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
    valid = np.arange(n_scales)  # all scales are valid except scale=1
    idx = valid[np.random.randint(valid.shape[0])]
    pts = scaled[idx]

    return pts


def get_translation_pts(hw_ratio, pts):
    # 平移范围
    t_min, t_max = np.min(pts - [-1., -hw_ratio], axis=0), np.min([1., hw_ratio] - pts, axis=0)
    pts += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                    np.random.uniform(-t_min[1], t_max[1])]),
                          axis=0)

    return pts


def get_perspective_pts(hw_ratio, pts, perspective_amplitude):
    # 获取正态分布的增幅，我觉得不好，这样子趋于均匀了
    perspective_amplitude_x = np.random.normal(0., perspective_amplitude / 2, 2)
    perspective_amplitude_y = np.random.normal(0., hw_ratio * perspective_amplitude / 2, 2)

    # perspective_amplitude_x = (np.random.rand(2) - 0.5) * perspective_amplitude
    # perspective_amplitude_y = (np.random.rand(2) - 0.5) * hw_ratio * perspective_amplitude

    # 限制振幅，标准差也就拿来用用
    perspective_amplitude_x = np.clip(perspective_amplitude_x, -perspective_amplitude / 2,
                                      perspective_amplitude / 2)
    perspective_amplitude_y = np.clip(perspective_amplitude_y, hw_ratio * -perspective_amplitude / 2,
                                      hw_ratio * perspective_amplitude / 2)

    # 根据不同方向挪动。。。左上 左下 右上 右下，都是向外扩张了
    pts[0, 0] -= perspective_amplitude_x[1]  # 其实有正有负
    pts[0, 1] -= perspective_amplitude_y[1]

    pts[1, 0] -= perspective_amplitude_x[0]
    pts[1, 1] += perspective_amplitude_y[1]

    pts[2, 0] += perspective_amplitude_x[1]
    pts[2, 1] -= perspective_amplitude_y[0]

    pts[3, 0] += perspective_amplitude_x[0]
    pts[3, 1] += perspective_amplitude_y[0]

    return pts


def get_fixed_h(shape, r, theta, img):
    # 理论上没问题的，因为最终都是在ToTensor的基础上进行的
    width = float(shape[1])
    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
    pts2 = pts1.copy() * r
    pts2[:, 1] *= hw_ratio

    center = np.mean(pts2, axis=0, keepdims=True)
    rot_mat = np.reshape(np.stack([np.cos(theta), -np.sin(theta), np.sin(theta),
                                   np.cos(theta)], axis=0), [-1, 2, 2])

    # 获取旋转后的点 np里的 tile 和 torch里的 repeat 是一样的样子
    rotated = np.matmul(np.tile(np.expand_dims(pts2 - center, axis=0), [1, 1, 1]), rot_mat) + center

    # 筛选在范围内的点
    pts2 = rotated[0]

    pts2[:, 1] /= hw_ratio

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    homography = np.concatenate([homography, [1.]]).reshape(3, 3)

    device = img.device
    homography = torch.from_numpy(homography).float().to(device)  # 转torch
    img_warped = get_aug_img(img, homography)

    return img, img_warped, homography.unsqueeze(0)


def sample_homography(
        shape,
        order=None,
        patch_ratio=0.8,
        perspective_amplitude=0.4,
        n_scales=100, scaling_amplitude=0.1,
        n_angles=100, max_angle=pi / 4, min_angle=None, angle_type=0):
    """ Sample a random homography that includes perspective, scale, translation and rotation operations."""

    if order is None:
        order = ['perspective', 'scale', 'translation', 'rotation']
    width = float(shape[1])
    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
    pts2 = pts1.copy() * patch_ratio
    pts2[:, 1] *= hw_ratio

    for key in order:
        if key == 'perspective':
            pts2 = get_perspective_pts(hw_ratio, copy.deepcopy(pts2), perspective_amplitude)

        elif key == 'scale':
            pts2 = get_scale_pts(hw_ratio, copy.deepcopy(pts2), scaling_amplitude, n_scales)

        elif key == 'translation':
            pts2 = get_translation_pts(hw_ratio, copy.deepcopy(pts2))

        elif key == 'rotation':
            pts2 = get_rotate_pts(hw_ratio, copy.deepcopy(pts2), min_angle, max_angle, n_angles, angle_type)

    pts2[:, 1] /= hw_ratio

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    homography = np.concatenate([homography, [1.]]).reshape(3, 3)
    return homography


def warp_homography(sources, homography):
    """Warp features given a homography

    Parameters
    ----------
    sources: torch.tensor (1,H,W,2)
        Keypoint vector.
    homography: torch.Tensor (3,3)
        Homography.

    Returns
    -------
    warped_sources: torch.tensor (1,H,W,2)
        Warped feature vector.
    """
    _, H, W, _ = sources.shape
    warped_sources = sources.clone().squeeze()
    warped_sources = warped_sources.view(-1, 2)
    warped_sources = torch.addmm(homography[:, 2], warped_sources, homography[:, :2].t())
    warped_sources.mul_(1 / warped_sources[:, 2].unsqueeze(1))
    warped_sources = warped_sources[:, :2].contiguous().view(1, H, W, 2)
    return warped_sources


def add_noise(img, mode="gaussian", percent=0.02):
    """Add image noise

    Parameters
    ----------
    image : np.array
        Input image
    mode: str
        Type of noise, from ['gaussian','salt','pepper','s&p']
    percent: float
        Percentage image points to add noise to.
    Returns
    -------
    image : np.array
        Image plus noise.
    """
    original_dtype = img.dtype
    if mode == "gaussian":
        mean = 0
        var = 0.1
        sigma = var * 0.5

        if img.ndim == 2:
            h, w = img.shape
            gauss = np.random.normal(mean, sigma, (h, w))
        else:
            h, w, c = img.shape
            gauss = np.random.normal(mean, sigma, (h, w, c))

        if img.dtype not in [np.float32, np.float64]:
            gauss = gauss * np.iinfo(img.dtype).max
            img = np.clip(img.astype(np.float) + gauss, 0, np.iinfo(img.dtype).max)
        else:
            img = np.clip(img.astype(np.float) + gauss, 0, 1)

    elif mode == "salt":
        print(img.dtype)
        s_vs_p = 1
        num_salt = np.ceil(percent * img.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in img.shape])

        if img.dtype in [np.float32, np.float64]:
            img[coords] = 1
        else:
            img[coords] = np.iinfo(img.dtype).max
            print(img.dtype)
    elif mode == "pepper":
        s_vs_p = 0
        num_pepper = np.ceil(percent * img.size * (1.0 - s_vs_p))
        coords = tuple(
            [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        )
        img[coords] = 0

    elif mode == "s&p":
        s_vs_p = 0.5

        # Salt mode
        num_salt = np.ceil(percent * img.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in img.shape])
        if img.dtype in [np.float32, np.float64]:
            img[coords] = 1
        else:
            img[coords] = np.iinfo(img.dtype).max

        # Pepper mode
        num_pepper = np.ceil(percent * img.size * (1.0 - s_vs_p))
        coords = tuple(
            [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        )
        img[coords] = 0
    else:
        raise ValueError("not support mode for {}".format(mode))

    noisy = img.astype(original_dtype)
    return noisy


def non_spatial_augmentation(img_warp_ori, jitter_parameters=None, color_order=None,
                             to_gray=False, if_add_noise=True, if_add_blur=True, if_add_rec=False):
    """ Apply non-spatial augmentation to an image (jittering, color swap, convert to gray scale, Gaussian blur)."""

    # pycharm编译器建议我这么干的。。。
    if jitter_parameters is None:
        jitter_parameters = [0, 0, 0, 0]
    if color_order is None:
        color_order = [0, 1, 2]

    brightness, contrast, saturation, hue = jitter_parameters
    color_augmentation = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)

    B = img_warp_ori.shape[0]
    img_warp = []
    kernel_sizes = [0, 1, 3, 5]
    for b in range(B):
        img_warp_sub = img_warp_ori[b].cpu()
        img_warp_sub = torchvision.transforms.functional.to_pil_image(img_warp_sub)

        img_warp_sub_np = np.array(img_warp_sub)

        if if_add_rec:
            h, w, c = img_warp_sub_np.shape
            rec_num = np.random.randint(40, 60)
            hehe = np.zeros((h, w, c), dtype='uint8')
            for i in range(rec_num):
                rand_x = np.random.randint(0, w)
                rand_y = np.random.randint(0, h)
                rand_w = np.random.randint(w // 40, w // 8)
                rand_h = np.random.randint(h // 40, h // 8)
                rand_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
                if np.random.rand() > 0.5:
                    cv2.rectangle(hehe, (rand_x, rand_y), (rand_x + rand_w, rand_y + rand_h), rand_color, -1)
                else:
                    cv2.circle(hehe, (rand_x, rand_y), int(0.5 * (rand_w + rand_h)), rand_color, -1)

            alpha = 0.1 * np.random.rand()
            if np.random.rand() > 0.5:
                hehe = 255 - hehe
            img_warp_sub_np = cv2.addWeighted(hehe, alpha, img_warp_sub_np, 1 - alpha, 0)

        img_warp_sub_np = img_warp_sub_np[:, :, color_order]

        if if_add_noise:
            if np.random.rand() > 0.5:
                img_warp_sub_np = add_noise(img_warp_sub_np)

        rand_index = np.random.randint(4)
        kernel_size = kernel_sizes[rand_index]
        if if_add_blur:
            if kernel_size > 0:
                img_warp_sub_np = cv2.GaussianBlur(img_warp_sub_np, (kernel_size, kernel_size), sigmaX=0)

        if to_gray:
            img_warp_sub_np = cv2.cvtColor(img_warp_sub_np, cv2.COLOR_RGB2GRAY)
            img_warp_sub_np = cv2.cvtColor(img_warp_sub_np, cv2.COLOR_GRAY2RGB)

        img_warp_sub = Image.fromarray(img_warp_sub_np)
        img_warp_sub = color_augmentation(img_warp_sub)

        img_warp_sub = torchvision.transforms.functional.to_tensor(img_warp_sub).to(img_warp_ori.device)

        img_warp.append(img_warp_sub)

    img_warp = torch.stack(img_warp, dim=0)
    return img_warp


def my_show_img(img):
    """将训练过程中的归一化tensor图片显示出来"""
    img1 = (img.squeeze().permute(1, 2, 0).detach().cpu().clone().squeeze())
    # img1 -= img1.min()
    # img1 /= img1.max()
    img1 = (img1 * 255).numpy().astype(np.uint8)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.show()


def tensorimg2cv2(img):
    # toTensor只是除以255罢了
    # 这个转换基本一样的
    img1 = (img.squeeze().permute(1, 2, 0).detach().cpu().clone().squeeze())
    # img1 -= img1.min()
    # img1 /= img1.max()
    img1 = (img1 * 255).numpy().astype(np.uint8)  # 不一定是255
    return img1


def cv22tensorimg(img):
    img1 = Image.fromarray(img)
    transform = transforms.ToTensor()
    img1 = transform(img1).type('torch.FloatTensor')
    return img1


def get_patch(coord_norm, input_img, size=32):
    """
    没有batch维度
    Parameters
    ----------
    coord_norm: tensor
        N * 2
    input_img
    size

    Returns
    -------

    """
    with torch.no_grad():  # 我就加着玩玩，虽然加不加一样的
        _, H, W = input_img.shape
        N = coord_norm.numel() // 2  # (x,y)嘛

        # 周围填充，后面计算偏移的
        img = F.pad(input_img.clone().detach().unsqueeze(0),
                    (size // 2, size // 2, size // 2, size // 2),
                    'constant', 0).repeat(N, 1, 1, 1)  # 加个 N 维度

        # 获取size大小正常归一化网格点，后面我要直接加法做偏移的
        coord_grid = image_grid(N, H + size, W + size,
                                dtype=coord_norm.dtype,
                                device=coord_norm.device,
                                ones=False, normalized=True)[:, :, :size, :size]  # N C H W 插值还要变成 N H W C格式的

        # coord_norm 转偏置，随后与网格点相加，获取真正网格点，然后变成 N H W C，进行插值，获取 N 个patches --> N 3 size size
        coord_grid.add_((coord_norm + 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size, size))
        coord_grid.permute_(0, 2, 3, 1)
        patches = torch.nn.functional.grid_sample(img, coord_grid, align_corners=False)

    return patches  # 这里 patches.requires_grad = False


def get_aug_img(img, homography):
    _, _, H, W = img.shape
    # 获取网格点，并座单应性变换
    grid = image_grid(1, H, W,
                      dtype=img.dtype,
                      device=img.device,
                      ones=False, normalized=True).clone().permute(0, 2, 3, 1)
    grid_warped = warp_homography(grid, homography)
    aug_img = torch.nn.functional.grid_sample(img, grid_warped, align_corners=True)

    return aug_img


def set_img_jitter(img_warp_ori, jitter_parameters=None):
    if jitter_parameters is None:
        jitter_parameters = [0.5, 0.5, 0.2, 0.05]
    brightness, contrast, saturation, hue = jitter_parameters
    color_augmentation = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)
    B = img_warp_ori.shape[0]
    img_warp = []
    for b in range(B):
        img_warp_sub = img_warp_ori[b].cpu()
        img_warp_sub = torchvision.transforms.functional.to_pil_image(img_warp_sub)
        img_warp_sub = color_augmentation(img_warp_sub)
        img_warp_sub = torchvision.transforms.functional.to_tensor(img_warp_sub).to(img_warp_ori.device)
        img_warp.append(img_warp_sub)

    img_warp = torch.stack(img_warp, dim=0)
    return img_warp


def ha_augment_sample(data, jitter_parameters=None, aug_type='hard'):
    """Apply Homography Adaptation image augmentation."""

    if jitter_parameters is None:
        jitter_parameters = [0.5, 0.5, 0.2, 0.05]

    target_img = data['image'].clone().unsqueeze(0)  # 这里的image是未变化的
    del data['image']
    _, _, H, W = target_img.shape
    device = target_img.device

    # 空间变换 --------------------------------------------------------------------------------------
    if aug_type == 'hard':
        homography = sample_homography([H, W],  # 形变简单化
                                       ['perspective', 'translation', 'rotation'],
                                       patch_ratio=0.665,
                                       perspective_amplitude=0.57,
                                       max_angle=pi / 4)

    elif aug_type == 'easy':
        jitter_parameters = (0.2, 0.2, 0.2, 0.05)  # 色彩简单化
        homography = sample_homography([H, W],  # 形变简单化
                                       patch_ratio=0.7,
                                       scaling_amplitude=0.2,
                                       perspective_amplitude=0.4,
                                       max_angle=pi / 4)

    else:
        print("fuck you!!!\r\nhaha")

    homography = torch.from_numpy(homography).float().to(device)  # 转torch
    source_img = get_aug_img(target_img, homography)

    # 色彩、对比度、噪声、滤波等等 --------------------------------------------------------------------------------------
    color_order = [0, 1, 2]
    cao = np.random.rand(2)
    if cao[0] > 0.5:
        random.shuffle(color_order)
    to_gray = False
    if cao[1] > 0.5:
        to_gray = True

    target_img = non_spatial_augmentation(target_img,
                                          jitter_parameters=jitter_parameters,
                                          color_order=color_order,
                                          to_gray=to_gray,
                                          if_add_noise=False)

    source_img = non_spatial_augmentation(source_img,
                                          jitter_parameters=jitter_parameters,
                                          color_order=color_order,
                                          to_gray=to_gray,
                                          if_add_noise=False)
    data['img'] = target_img.squeeze()
    data['img_aug'] = source_img.squeeze()
    data['homography'] = homography
    # my_show_img(data['img'])
    # my_show_img(data['img_aug'])
    return data


def get_warp_point(point, HM):
    # point是?*2矩阵 W = 320, H = 256
    point_warp = copy.deepcopy(point)
    point_warp[:, 0] = (HM[0, 0] * point[:, 0] + HM[0, 1] * point[:, 1] + HM[0, 2]) / \
                       (HM[2, 0] * point[:, 0] + HM[2, 1] * point[:, 1] + HM[2, 2])
    point_warp[:, 1] = (HM[1, 0] * point[:, 0] + HM[1, 1] * point[:, 1] + HM[1, 2]) / \
                       (HM[2, 0] * point[:, 0] + HM[2, 1] * point[:, 1] + HM[2, 2])

    return point_warp


def get_norm_point(point):
    # point是?*2矩阵 W = 320, H = 256
    point_norm = copy.deepcopy(point)
    point_norm[:, 0] = point[:, 0] / 159.5 - 1
    point_norm[:, 1] = point[:, 1] / 127.5 - 1
    return point_norm


def get_unnorm_point(point_norm):
    # point是?*2矩阵 W = 320, H = 256
    point = copy.deepcopy(point_norm)
    point[:, 0] = (point_norm[:, 0] + 1) * 159.5
    point[:, 1] = (point_norm[:, 1] + 1) * 127.5
    return point
