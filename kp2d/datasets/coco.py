# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob
from PIL import Image
from torch.utils.data import Dataset
import random


class COCOLoader(Dataset):
    """
    Coco dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    data_transform : Function
        Transformations applied to the sample
    """

    def __init__(self, root_dir, data_transform=None, data_size=500000):

        super().__init__()
        self.root_dir = root_dir

        self.files = []

        self.limited = data_size  # 限制张数嘛

        for filename in glob.glob(root_dir + '/*.jpg'):
            self.files.append(filename)

        self.data_transform = data_transform
        self._aug_type = 'hard'  # 默认hard

    def set_aug_type(self, aug_type='hard'):
        self._aug_type = aug_type

    def __len__(self):
        return min(len(self.files), self.limited)

    def _read_rgb_file(self, filename):
        return Image.open(filename)

    def __getitem__(self, idx):

        filename = self.files[idx]
        image = self._read_rgb_file(filename)

        if image.mode == 'L':
            image_new = Image.new("RGB", image.size)
            image_new.paste(image)
            sample = {'image': image_new, 'idx': idx}
        else:
            sample = {'image': image, 'idx': idx}

        if self.data_transform:
            sample = self.data_transform(sample, self._aug_type)

        return sample
