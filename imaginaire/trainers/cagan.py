# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os

import numpy as np
import torch
from torch import nn

from imaginaire.evaluation import compute_fid
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import is_master


def get_xi_yi_yj(batch):
    return batch[:, :3, :, :], batch[:, 3:6, :, :], batch[:, 6:, :, :]


def get_alpha_xij(out_tensor):
    return out_tensor[:, 0:1, :, :], out_tensor[:, 1:, :, :]


class Trainer(BaseTrainer):
    r"""Reimplementation of the CAGAN ( https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w32/Jetchev_The_
Conditional_Analogy_ICCV_2017_paper.pdf)
    algorithm.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader):
        super().__init__(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                         train_data_loader, val_data_loader)

        self.use_cuda = cfg.trainer.use_cuda

    def _init_loss(self, cfg):
        r"""Initialize loss terms. In FUNIT, we have several loss terms
        including the GAN loss, the image reconstruction loss, the feature
        matching loss, and the gradient penalty loss.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria['gen_basic'] = nn.BCELoss()
        self.criteria['dis_basic'] = nn.BCELoss()
        self.criteria['clycle_loss'] = nn.L1Loss()

        for loss_name, loss_weight in cfg.trainer.loss_weight.__dict__.items():
            if loss_weight > 0:
                self.weights[loss_name] = loss_weight
        pass

    def construct(self, xi, yi, yj):
        data = torch.cat([xi, yi, yj], dim=1)
        fake_out = self.net_G(data)
        alpha, xij_temp = get_alpha_xij(fake_out)
        xij = alpha * xij_temp + (1 - alpha) * xi
        return xij, alpha, xij_temp

    def gen_basic_loss(self, x, y):
        dis_out = self.net_D(torch.cat([x, y], dim=1))
        label = torch.ones_like(dis_out)
        if self.use_cuda:
            label = label.cuda()
        loss = self.criteria['gen_basic'](dis_out, label)
        return loss

    def dis_basic_loss(self, x, y, is_pos):
        dis_out = self.net_D(torch.cat([x, y], dim=1))
        if is_pos:
            label = torch.ones_like(dis_out)
        else:
            label = torch.zeros_like(dis_out)
        if self.use_cuda:
            label = label.cuda()
        loss = self.criteria['dis_basic'](dis_out, label)
        return loss

    def gen_loss(self, g_fake_xij, f_fake_xij, yi, yj, xi, g_fake_alpha, f_fake_alpha):
        self.gen_losses['g_fake'] = self.gen_basic_loss(g_fake_xij, yj)
        self.gen_losses['f_fake'] = self.gen_basic_loss(f_fake_xij, yi)
        self.gen_losses['cycle'] = self.criteria['clycle_loss'](f_fake_xij, xi)
        # self.gen_losses['id'] = torch.norm(g_fake_alpha, p=1) + torch.norm(f_fake_alpha, p=1)
        loss_sum = self._get_total_loss(gen_forward=True)
        # print("gen losses: {}, sum: {}".format(self.gen_losses, loss_sum))
        return loss_sum

    def dis_loss(self, xi, yi, xij, yj):
        self.dis_losses["real"] = self.dis_basic_loss(xi, yi, is_pos=True)
        self.dis_losses["fake"] = self.dis_basic_loss(xi, yj, is_pos=False) + \
                                  self.dis_basic_loss(xij, yj, is_pos=False)
        loss_sum = self._get_total_loss(gen_forward=False)
        # print("dis losses: {}, sum: {}".format(self.dis_losses, loss_sum))
        return loss_sum

    def gen_forward(self, data):
        r"""Compute the loss for FUNIT generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        xi, yi, yj = get_xi_yi_yj(data)
        # generator forward
        g_fake_xij, g_fake_alpha,  g_fake_xij_temp = self.construct(xi, yi, yj)
        # generator cycle
        f_fake_xij, f_fake_alpha, f_fake_xij_temp = self.construct(g_fake_xij, yj, yi)
        # Compute total loss
        total_loss = self.gen_loss(f_fake_xij, f_fake_xij, yi, yj, xi, g_fake_alpha, f_fake_alpha)
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for FUNIT discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        # generator forward
        xi, yi, yj = get_xi_yi_yj(data)
        with torch.no_grad():
            g_fake_xij, g_fake_alpha, g_fake_xij_temp = self.construct(xi, yi, yj)
        g_fake_xij.requires_grad = True
        total_loss = self.dis_loss(xi, yi, g_fake_xij, yj)
        return total_loss

    def _get_visualizations(self, data):
        r"""Compute visualization image.

        Args:
            data (dict): The current batch.
        """
        with torch.no_grad():
            xi, yi, yj = get_xi_yi_yj(data)
            # generator forward
            g_fake_xij, g_fake_alpha, g_fake_xij_temp = self.construct(xi, yi, yj)
            g_fake_xij_ = g_fake_xij * 0.5 + 0.5
            xi_ = xi * 0.5 + 0.5
            yj_ = yj * 0.5 + 0.5
            vis_images = [xi_, yj_, g_fake_xij_]
            return vis_images

    def _compute_fid(self):
        r"""Compute FID. We will compute a FID value per test class. That is
        if you have 30 test classes, we will compute 30 different FID values.
        We will then report the mean of the FID values as the final
        performance number as described in the FUNIT paper.
        """
        self.net_G.eval()
        all_fid_values = []
        fid_path = self._get_save_path("valid", 'npy')
        fid_value = compute_fid(fid_path, self.val_data_loader,
                                self.net_G, 'images_style',
                                'images_trans')
        all_fid_values.append(fid_value)

        if is_master():
            mean_fid = np.mean(all_fid_values)
            print('Epoch {:05}, Iteration {:09}, Mean FID {}'.format(
                self.current_epoch, self.current_iteration, mean_fid))
            return mean_fid
        else:
            return None
