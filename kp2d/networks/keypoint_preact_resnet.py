# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from kp2d.utils.keypoints import get_border_mask, get_uv_norm
from kp2d.utils.image import image_grid
from kp2d.networks.preact_resnet import PreActResNet18, PreActResNet34
from kp2d.networks.netvlad import NetVLAD


class Decoder(nn.Module):
    """ D-LinkNet里解码（转置卷积）的过程，包含三个步骤 """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        # 反卷积参数kernel_size=3,stride=2,padding=1,尺寸放大一倍
        # 反卷积参数kernel_size=3,stride=1,padding=1,尺寸不变
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, in_planes, kernel_size, stride,
                               padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True), )

    def forward(self, x):
        x = self.tp_conv(x)
        return x


class Center(nn.Module):
    """ 空洞卷积，为了包含更多的感受野 """

    def __init__(self, in_planes, out_planes, dilation=0, bias=False):
        super(Center, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, 3, 1, dilation, dilation, bias=bias),
                                  nn.BatchNorm2d(out_planes),
                                  nn.ReLU(inplace=True), )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class KeypointEncoder(nn.Module):
    def __init__(self, pretrained, with_drop):
        super(KeypointEncoder, self).__init__()

        self.rn = PreActResNet18()

        self.dropout = nn.Dropout2d(0.2)
        self.use_dropout = with_drop

    def forward(self, input_image):
        x = self.rn.relu(self.rn.bn1(self.rn.conv1(input_image)))
        l1 = self.rn.layer1(self.rn.maxpool(x)) if not self.use_dropout else self.dropout(
            self.rn.layer1(self.rn.maxpool(x)))
        l2 = self.rn.layer2(l1) if not self.use_dropout else self.dropout(self.rn.layer2(l1))
        l3 = self.rn.layer3(l2) if not self.use_dropout else self.dropout(self.rn.layer3(l2))
        l4 = self.rn.layer4(l3) if not self.use_dropout else self.dropout(self.rn.layer4(l3))

        return [x, l1, l2, l3, l4]


class KeypointDecoder(nn.Module):
    def __init__(self):
        super(KeypointDecoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_dec = np.array([32, 64, 128, 256, 256])

        # Layer4
        self.upconv4_0 = ConvBnRelu(self.num_ch_enc[4], self.num_ch_dec[4])  # 512 256
        self.upconv4_1 = ConvBnRelu(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])  # 512 256

        # Layer3
        self.upconv3_0 = ConvBnRelu(self.num_ch_dec[4], self.num_ch_dec[3])  # 256 256
        self.upconv3_1 = ConvBnRelu(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])  # 384 256

        # Layer2
        self.upconv2_0 = ConvBnRelu(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv2_1 = ConvBnRelu(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])

        # Layer1
        self.upconv1_0 = ConvBnRelu(self.num_ch_dec[2], self.num_ch_dec[1])
        self.upconv1_1 = ConvBnRelu(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])

        # 中间部分，膨胀卷积
        self.center_a1 = Center(512, 512, 1)
        self.center_a2 = Center(512, 512, 2)
        self.center_a3 = Center(512, 512, 4)
        self.center_b1 = Center(512, 512, 1)
        self.center_b2 = Center(512, 512, 2)
        self.center_c1 = Center(512, 512, 1)

        self.decoder1 = Decoder(64, 64, 3, 2, 1, 1)
        self.decoder2 = Decoder(128, 128, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 256, 3, 2, 1, 1)
        self.decoder4 = Decoder(256, 256, 3, 2, 1, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = {}

        # -------------------- center --------------------
        c_a1 = self.center_a1(input_features[4])
        c_a2 = self.center_a2(c_a1)
        c1 = self.center_a3(c_a2)
        c_b1 = self.center_b1(input_features[4])
        c2 = self.center_b2(c_b1)
        c3 = self.center_c1(input_features[4])

        # -------------------- decoder --------------------
        x = input_features[4] + c1 + c2 + c3
        # Layer4
        x = self.upconv4_0(x)  # 512-256  /32
        x = [self.decoder4(x)]  # 256 /16
        x += [input_features[3]]  # 256-512 /16
        x = torch.cat(x, 1)  # 512 /16
        x = self.upconv4_1(x)  # 512-256 /16
        # Layer3
        x = self.upconv3_0(x)  # 256-256 /16
        x = [self.decoder3(x)]  # 256 /8
        x += [input_features[2]]  # 256-384 /8
        x = torch.cat(x, 1)  # 384 /8
        x = self.upconv3_1(x)  # 384-256 /8
        # Detector and score
        outputs['feature_map_detect'] = x.clone()
        # Layer2
        x = self.upconv2_0(x)  # 256-128 /8
        x = [self.decoder2(x)]  # 128 /4
        x += [input_features[1]]  # 128-192 /4
        x = torch.cat(x, 1)  # 192 /4
        x = self.upconv2_1(x)  # 192-128 /4
        # Layer1
        x = self.upconv1_0(x)  # 128-64 /4
        x = [self.decoder1(x)]  # 64 /2
        x += [input_features[0]]  # 128 /2
        x = torch.cat(x, 1)
        x = self.upconv1_1(x)  # 128-64 /2  # 这么精细的嘛，维度这么低准确率能高？
        # Descriptor features
        outputs['feature_map_desc'] = x

        return outputs


class KeypointHeader(nn.Module):
    """ 单独一个 header 拿出来方便 """

    def __init__(self):
        super(KeypointHeader, self).__init__()
        self.detector_nbr = 256
        self.descriptor_nbr = 64
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.pad = nn.ReflectionPad2d(1)

        # Score
        self.score_conv = nn.Sequential(
            nn.Conv2d(self.detector_nbr, self.detector_nbr, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.detector_nbr),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.detector_nbr, 1, 3))
        # Detector
        self.loc_conv = nn.Sequential(
            nn.Conv2d(self.detector_nbr, self.detector_nbr, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.detector_nbr),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.detector_nbr, 2, 3))
        # Descriptor
        self.desc_conv = nn.Sequential(
            nn.Conv2d(self.descriptor_nbr, self.descriptor_nbr, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.descriptor_nbr),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.descriptor_nbr, self.descriptor_nbr, 3))

    def init_weights(self):
        """ 暂时没用 """
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: dict):
        """

        Parameters
        ----------
        desc_type: list, contain str
            use which desc: 'perspective', 'rotate', 'scale'
        x: dict
            'feature_map_detect'
            'feature_map_desc'

        Returns
        -------

        """
        feature_map_detect = x['feature_map_detect']
        feature_map_desc = x['feature_map_desc']
        loc_output = self.tanh(self.loc_conv(feature_map_detect))
        score_output = self.sigmoid(self.score_conv(feature_map_detect))
        desc_output = self.desc_conv(feature_map_desc)

        return loc_output, score_output, desc_output


class KeypointPreactResnet(nn.Module):
    def __init__(self, with_drop=True):
        super().__init__()
        print('Instantiating keypoint preact resnet')

        # encoder, decoder and header
        self.encoder = KeypointEncoder(pretrained=True, with_drop=True)
        self.decoder = KeypointDecoder()
        self.header = KeypointHeader()
        self.netvlad = NetVLAD(num_clusters=32, dim=64)

        self.cross_ratio = 2.0
        self.cell = 8

    def forward(self, x, need_grid_desc=False, is_need_netvlad=False):
        B, _, H, W = x.shape

        # -------------------- get three headers --------------------
        if is_need_netvlad:
            with torch.no_grad():
                x = self.encoder(x)
                x = self.decoder(x)
                center_shift, score, desc = self.header(x)
        else:
            x = self.encoder(x)
            x = self.decoder(x)
            center_shift, score, desc = self.header(x)

        _, _, Hc, Wc = score.shape

        # -------------------- Remove border for score --------------------
        border_mask = get_border_mask(B, Hc, Wc, score.device)
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask

        # -------------------- Remap coordinate --------------------
        step = (self.cell - 1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_unnorm_ori = center_base.add(center_shift.mul(self.cross_ratio * step))  # 多用个变量使得梯度正常
        coord_unnorm = coord_unnorm_ori.clone()
        coord_unnorm[:, 0] = torch.clamp(coord_unnorm_ori[:, 0], min=0, max=W - 1)
        coord_unnorm[:, 1] = torch.clamp(coord_unnorm_ori[:, 1], min=0, max=H - 1)

        # -------------------- grid but not l2 norm --------------------
        coord_norm = get_uv_norm(coord_unnorm, H, W)  # 万一要用呢
        desc_grid = torch.nn.functional.grid_sample(desc, coord_norm, align_corners=False)
        desc_grid_l2norm = F.normalize(desc_grid, p=2, dim=1)

        netvlad_desc = None
        if is_need_netvlad:
            netvlad_desc = F.normalize(self.netvlad(desc_grid_l2norm), p=2, dim=1)
        # -------------------- Get desc, if training, return l2 norm desc --------------------
        if self.training is False:
            desc = desc_grid_l2norm
        if is_need_netvlad:
            return score, coord_unnorm, desc, netvlad_desc
        if need_grid_desc:
            return score, coord_unnorm, desc, desc_grid_l2norm
        return score, coord_unnorm, desc
