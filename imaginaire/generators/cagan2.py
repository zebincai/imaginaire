import torch
import torch.nn as nn
from imaginaire.layers import Conv2dBlock


class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, stride=1, padding=1, use_norm=True, use_act=True):
        super(UpscaleBlock, self).__init__()
        self.use_norm = use_norm
        self.use_act = use_act
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        if self.use_norm:
            self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=True)
        if self.use_act:
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, x2):
        out = self.upsample(x)
        out = torch.cat([out, x2], dim=1)
        out = self.conv(out)
        if self.use_norm:
            out = self.norm(out)
        if self.use_act:
            out = self.activation(out)
        return out


class Conv2dSame(nn.Module):
    """ Applies a 2D convolution  over an input signal composed of several input
        planes and output size is the same as input size.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (only tuple): Size of the convolving kernel
            stride (only int): Stride of the convolution. Default: 1
            padding_layer: type of layer for padding. Default: nn.ZeroPad2d
            dilation (only int): Spacing between kernel elements. Default: 1
            bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding_layer=nn.ZeroPad2d, dilation=1, bias=True):
        super().__init__()

        kernel_h, kernel_w = kernel_size

        k = dilation * (kernel_w - 1) - stride + 2
        pr = k // 2
        pl = pr - 1 if k % 2 == 0 else pr

        k = dilation * (kernel_h - 1) - stride + 1 + 1
        pb = k // 2
        pt = pb - 1 if k % 2 == 0 else pb
        self.pad_same = padding_layer((pl, pr, pb, pt))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.pad_same(x)
        x = self.conv(x)
        return x


# class Out_Branch(nn.Module):
#     def __init__(self, nc_in, kernel_size=(4, 3)):
#         super(Out_Branch, self).__init__()
#         #self.conv = Conv2dSame(nc_in, 4, kernel_size, bias=False)
#         self.conv = nn.Conv2d(nc_in, 4,  3, padding=1)
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x = self.conv(x)
#         alpha = x[:, 0:1, :, :]
#         x_i_j = x[:, 1:, :, :]
#         alpha = self.sigmoid(alpha)
#         x_i_j = self.tanh(x_i_j)
#         return torch.cat([alpha, x_i_j], 1)
       
        #return x

class Out_Branch(nn.Module):
    def __init__(self, nc_in, kernel_size=(4, 3)):
        super(Out_Branch, self).__init__()
        #self.conv = Conv2dSame(nc_in, 4, kernel_size, bias=False)
        self.conv = nn.Conv2d(nc_in, 4,  3, padding=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        # out = self.sigmoid(out)
        return out


class Generator(nn.Module):
    r"""Dummy generator.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super(Generator, self).__init__()
        # input downsample
        nonlinearity = gen_cfg.nonlinearity
        self.downsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.downsample3 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=0.0625, mode='bilinear', align_corners=True)

        conv_params = dict(kernel_size=3,
                           padding=1,
                           activation_norm_type="instance",
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True)
        # encoder
        self.layer1 = Conv2dBlock(in_channels=9, out_channels=64, kernel_size=3, padding=1, stride=2,
                                  nonlinearity=nonlinearity, inplace_nonlinearity=True)
        self.layer2 = Conv2dBlock(in_channels=64+6, out_channels=128, stride=2, **conv_params)
        self.layer3 = Conv2dBlock(in_channels=128+6, out_channels=256, stride=2, **conv_params)
        self.layer4 = Conv2dBlock(in_channels=256+6, out_channels=512, stride=2, **conv_params)

        # decoder
        self.layer5 = UpscaleBlock(in_channels=512+256+6+6, out_channels=256)
        self.layer6 = UpscaleBlock(in_channels=256+128 + 6, out_channels=128)
        self.layer7 = UpscaleBlock(in_channels=128+64 + 6, out_channels=64)
        self.layer8 = UpscaleBlock(in_channels=64 +6, out_channels=64, use_norm=True, use_act=True)
        self.outlayer = Out_Branch(nc_in=64)

    def forward(self, x):
        r"""Dummy Generator forward.
        Args:
            data (tensor):
        """
        xi = x[:, 0:3, :, :]
        yj = x[:, 6:, :, :]
        xi_yj = torch.cat([xi, yj], dim=1)
        x_d02 = self.downsample1(xi_yj)
        x_d04 = self.downsample2(xi_yj)
        x_d08 = self.downsample3(xi_yj)
        x_d16 = self.downsample4(xi_yj)

        # encoder
        x_en2 = self.layer1(x)
        x_en2 = torch.cat([x_en2, x_d02], dim=1)
        x_en4 = self.layer2(x_en2)
        x_en4 = torch.cat([x_en4, x_d04], dim=1)
        x_en8 = self.layer3(x_en4)
        x_en8 = torch.cat([x_en8, x_d08], dim=1)
        x_en16 = self.layer4(x_en8)
        x_en16 = torch.cat([x_en16, x_d16], dim=1)

        # decoder
        x_de8 = self.layer5(x_en16, x_en8)
        # x_de8 = torch.cat([x_de8, x_en8], dim=1)
        x_de4 = self.layer6(x_de8, x_en4)
        # x_de4 = torch.cat([x_de4, x_en4], dim=1)
        x_de2 = self.layer7(x_de4, x_en2)
        # x_de2 = torch.cat([x_de2, x_en2], dim=1)
        out = self.layer8(x_de2, xi_yj)
        out = self.outlayer(out)
        return out


if __name__ == "__main__":
    from imaginaire.config import Config
    cfg = Config("/configs/projects/cagan/LipMPV/base_dis2_gen1.yaml")
    gen = Generator(cfg.gen, cfg.data)
    batch = torch.randn((8, 9, 256, 192))
    y = gen(batch)
    print(y.shape)
