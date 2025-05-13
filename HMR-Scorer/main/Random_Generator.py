import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.smpler_x import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet
from nets.loss import CoordLoss, ParamLoss, CELoss
from utils.human_models import smpl_x
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
import copy
from mmpose.models import build_posenet
from mmcv import Config

class Random_Generator(nn.Module):
    def __init__(self):
        super(Random_Generator, self).__init__()
        self.multi_n = cfg.num_hypos[0]
        self.noise_var_shape = getattr(cfg, 'noise_var_shape', 0.05)             # computed from test error distribution on ARCTIC
        self.noise_var_cam = getattr(cfg, 'noise_var_cam', 0.)
        self.noise_var_expr = getattr(cfg, 'noise_var_expr', 0.01)
        self.noise_var_pose = getattr(cfg, 'noise_var_pose', 0.05)
        self.noise_var_root_pose = getattr(cfg, 'noise_var_root_pose', 0.02)

    def forward(self, inputs, targets, meta_info, mode):
        shape = targets['smplx_shape'][:, None, :].clone().repeat(1, self.multi_n, 1)
        cam_trans = targets['smplx_cam_trans'][:, None, :].clone().repeat(1, self.multi_n, 1)                                  # [bs, 1, 3]
        expr = targets['smplx_expr'][:, None, :].clone().repeat(1, self.multi_n, 1)
        pose = targets['smplx_pose'][:, None, :].clone().repeat(1, self.multi_n, 1)                                            # [bs, 1, 159]

        shape_noised = shape + torch.randn_like(shape) * self.noise_var_shape
        cam_trans_noised = cam_trans + torch.randn_like(cam_trans) * self.noise_var_cam
        expr_noised = expr + torch.randn_like(expr) * self.noise_var_expr
        pose_noised = pose.clone()
        pose_noised[..., 3:] = pose[..., 3:] + torch.randn_like(pose[..., 3:]) * self.noise_var_pose
        pose_noised[..., :3] = pose[..., :3] + torch.randn_like(pose[..., :3]) * self.noise_var_root_pose

        out = {}
        out['smplx_shape'] = shape_noised
        out['smplx_expr'] = expr_noised
        out['smplx_pose'] = pose_noised
        out['cam_trans'] = cam_trans_noised

        return out


def get_random_generator(mode):
    model = Random_Generator()
    return model