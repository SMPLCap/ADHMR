from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from easydict import EasyDict as edict


class Integral(nn.Module):
    """integral layer: heatmaps to joint locations"""
    def __init__(self,cfg):
        super(Integral, self).__init__()
        self.config = cfg
        self.image_size = np.array(cfg.hrnet.image_size)

    def generate_integral_preds_2d(self, heatmaps):
        """
        heatmaps: [N, j, h, w]
        """
        assert isinstance(heatmaps, torch.Tensor), \
            'heatmaps should be torch.Tensor'
        assert heatmaps.dim() == 4, 'batch_heatmaps should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        height = heatmaps.shape[2]
        width = heatmaps.shape[3]

        # softmax
        if self.config.hrnet.integral.use_softmax:
            # heatmaps[heatmaps<0.1] = -1e9
            heatmaps = (heatmaps.reshape((batch_size, num_joints, -1))*self.config.hrnet.integral.alpha).double()
            heatmaps = F.softmax(heatmaps, 2)
            heatmaps = heatmaps.reshape((batch_size, num_joints, height, width)).float()
        elif self.config.hrnet.integral.use_pow:
            temp = self.config.hrnet.integral.pow_temp
            heatmaps[heatmaps<0.1] = 0
            heatmaps = torch.pow(heatmaps, 1/temp)
            heatmaps = torch.div(heatmaps, torch.sum(heatmaps, dim=(2, 3), dtype=torch.float32, keepdim=True))
        else:
            heatmaps = torch.div(heatmaps, torch.sum(heatmaps, dim=(2, 3), dtype=torch.float32, keepdim=True))
            if np.isnan(heatmaps.clone().detach().cpu().numpy()).any():
                print(heatmap_sum)
                assert 0, 'encounter NAN when processing heatmaps in the Integral layer'
        accu_x = heatmaps.sum(dim=2)  # [N, j, w]
        accu_y = heatmaps.sum(dim=3)  # [N, j, h]

        accu_x = accu_x * torch.arange(float(width)).cuda()
        accu_y = accu_y * torch.arange(float(height)).cuda()
        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)

        return accu_x, accu_y

    def generate_integral_hm(self, heatmaps):
        """
        heatmaps: [N, j, h, w]
        """
        assert isinstance(heatmaps, torch.Tensor), \
            'heatmaps should be torch.Tensor'
        assert heatmaps.dim() == 4, 'batch_heatmaps should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        height = heatmaps.shape[2]
        width = heatmaps.shape[3]

        # softmax
        if self.config.hrnet.integral.use_softmax:
            # heatmaps[heatmaps<0.1] = -1e9
            heatmaps = (heatmaps.reshape((batch_size, num_joints, -1))*self.config.hrnet.integral.alpha).double()
            heatmaps = F.softmax(heatmaps, 2)
            heatmaps = heatmaps.reshape((batch_size, num_joints, height, width)).float()
        elif self.config.hrnet.integral.use_pow:
            temp = self.config.hrnet.integral.pow_temp
            heatmaps[heatmaps<0.1] = 0
            heatmaps = torch.pow(heatmaps, 1/temp)
            heatmaps = torch.div(heatmaps, torch.sum(heatmaps, dim=(2, 3), dtype=torch.float32, keepdim=True))
        else:
            heatmaps = torch.div(heatmaps, torch.sum(heatmaps, dim=(2, 3), dtype=torch.float32, keepdim=True))
            if np.isnan(heatmaps.clone().detach().cpu().numpy()).any():
                print(heatmap_sum)
                assert 0, 'encounter NAN when processing heatmaps in the Integral layer'

        return heatmaps

    def forward(self, heatmaps):
        """
        heatmaps: [4*N, 20, 64, 64]
        return: 
        """

        num_joints = heatmaps.shape[1]
        hm_width = heatmaps.shape[2]
        hm_height = heatmaps.shape[3]

        x, y = self.generate_integral_preds_2d(heatmaps)
        x = x / float(hm_width) - 0.5
        y = y / float(hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)*2    # [4*N, 20, 2]
        return preds