import os
import glob
import random
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


class JoinRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img1), F.hflip(img2)
        return img1, img2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Dataset(data.Dataset):
    r"""Base class for image/video datasets.

    Args:
        cfg (Config object): Input config.
        is_inference (bool): Training if False, else validation.
        is_test (bool): Final test set after training and validation.
    """

    def __init__(self, cfg, is_inference=False, is_test=False):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.phase = None
        if is_test:
            self.phase = "test"
            self.cfgdata = self.cfg.test_data
            data_info = self.cfgdata.test
        else:
            self.cfgdata = self.cfg.data
            if is_inference:
                self.phase = "val"
                data_info = self.cfgdata.val
            else:
                self.phase = "train"
                data_info = self.cfgdata.train

        self.name = self.cfgdata.name
        self.human_files = self.get_filenames(os.path.join(data_info.roots, "*person_half_front.jpg"))
        self.cloth_files = self.get_filenames(os.path.join(data_info.roots, "*cloth_front.jpg"))
        self.file_indexes = list(range(len(self.human_files)))
        self.height = data_info.augmentations.resize_h
        self.width = data_info.augmentations.resize_w
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=(self.height, self.width)),
            ])
            self.random_flip = transforms.RandomHorizontalFlip()
            self.join_random_flip = JoinRandomHorizontalFlip()

    def get_filenames(self, file_pattern):
        filenames = glob.glob(file_pattern)
        filenames.sort()
        return filenames

    def __len__(self):
        return len(self.human_files)

    def __getitem__(self, index):
        xi_file = self.human_files[index]
        yi_file = self.cloth_files[index]
        # Load article picture y_j randomly
        yj_index = index
        while yj_index == index:
            yj_index = np.random.choice(self.file_indexes)

        yj_file = self.cloth_files[yj_index]
        yi_img = Image.open(yi_file).resize((self.width, self.height))
        yj_img = Image.open(yj_file).resize((self.width, self.height))
        if self.phase == 'train':
            xi_img = Image.open(xi_file).resize((self.width + 30, self.height + 40))
            xi_img = self.transform(xi_img)
            xi_img, yi_img = self.join_random_flip(xi_img, yi_img)
            yj_img = self.random_flip(yj_img)
        else:
            xi_img = Image.open(xi_file).resize((self.width, self.height))
        xi = self.normalize(xi_img)
        yi = self.normalize(yi_img)
        yj = self.normalize(yj_img)
        sample = torch.cat([xi, yi, yj], 0)
        return sample


if __name__ == "__main__":
    from imaginaire.config import Config
    cfg = Config("D:/workspace/develop/imaginaire/configs/projects/cagan/LipMPV/base.yaml")
    tempdataset = Dataset(cfg)
    print(len(tempdataset))
    for index, data in enumerate(tempdataset):
        print(index)
        print(data.shape)