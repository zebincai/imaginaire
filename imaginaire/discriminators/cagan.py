# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
import torch.nn as nn
from imaginaire.layers import Conv2dBlock
from imaginaire.layers.misc import ApplyNoise


class Discriminator(nn.Module):
    """Dummy Discriminator constructor.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Discriminator, self).__init__()
        nonlinearity = gen_cfg.nonlinearity
        # input downsample
        self.downsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.downsample3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.downsample4 = nn.Upsample(scale_factor=0.0625, mode='bilinear')

        conv_params = dict(kernel_size=3,
                           padding=1,
                           activation_norm_type="instance",
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True)
        # encoder
        self.apply_noise = ApplyNoise()
        self.layer1 = Conv2dBlock(in_channels=6, out_channels=64, kernel_size=3, padding=1, stride=2,
                                  nonlinearity=nonlinearity, inplace_nonlinearity=True)
        self.layer2 = Conv2dBlock(in_channels=64 + 6, out_channels=128, stride=2, **conv_params)
        self.layer3 = Conv2dBlock(in_channels=128 + 6, out_channels=256, stride=2, **conv_params)
        self.layer4 = Conv2dBlock(in_channels=256 + 6, out_channels=512, stride=2, **conv_params)
        self.outlayer = Conv2dBlock(in_channels=512 + 6, out_channels=1, kernel_size=3,
                                    nonlinearity="sigmoid")

    def forward(self, x):
        x = self.apply_noise(x)
        x_d02 = self.downsample1(x)
        x_d04 = self.downsample2(x)
        x_d08 = self.downsample3(x)
        x_d16 = self.downsample4(x)
        # encoder
        x_en2 = self.layer1(x)
        x_en2 = torch.cat([x_en2, x_d02], dim=1)
        x_en4 = self.layer2(x_en2)
        x_en4 = torch.cat([x_en4, x_d04], dim=1)
        x_en8 = self.layer3(x_en4)
        x_en8 = torch.cat([x_en8, x_d08], dim=1)
        x_en16 = self.layer4(x_en8)
        x_en16 = torch.cat([x_en16, x_d16], dim=1)
        out = self.outlayer(x_en16)
        return out


if __name__ == "__main__":

    dis = Discriminator()
    batch = torch.randn((8, 6, 256, 192))
    y = dis(batch)
    print(y.shape)



