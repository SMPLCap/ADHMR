import os
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import open3d as o3d
import numpy as np
from utils.human_models import smpl_x
from config import cfg

from utils.transforms import world2cam, cam2pixel, rigid_align, batch_rigid_align


class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, out, gt_index):
        loss = self.ce_loss(out, gt_index)
        return loss
    

def get_pairwise_comp_probs(batch_preds, batch_std_labels, sigma=None):
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij

def rankloss(score, metric, mask=None, sigma=None):
    gtrank = (-metric).argsort().argsort().float()
    pred, gt = get_pairwise_comp_probs(score, gtrank, sigma=sigma)
    rankloss = F.binary_cross_entropy(input=torch.triu(pred, diagonal=1),
                                     target=torch.triu(gt, diagonal=1), reduction='none')
    rankloss = torch.sum(rankloss, dim=(2, 1)) * mask
    return rankloss.mean()

def error_2_score(error, min=10, max=300):
    score = (max - error) / (max - min)
    score = torch.clip(score, min=0, max=1)
    return score

class ScoreLoss(nn.Module): 
    def __init__(self):
        super(ScoreLoss, self).__init__()
        self.weight_mpvpe = 1.
        self.weight_pa_mppve = 1.
        self.weight_mpjpe = 1.
        self.weight_pa_mpjpe = 1.
        self.weight_mpvpe_mse = 100
        self.weight_pa_mppve_mse = 100
        self.weight_mpjpe_mse = 100
        self.weight_pa_mpjpe_mse = 100

        self.sigma = 5
        self.loss_func = functools.partial(rankloss, sigma=self.sigma)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, score_raw, score, output, gt):
        '''
        gt['mesh']: [bs, 10475, 3]
        output['mesh']: [bs*multi_n, 10475, 3]

        Returns:
         eval_result: [bs, multi_n]
        '''
        eval_result = {}
        bs = gt['mesh'].shape[0]
        assert output['mesh'].shape[0] % bs == 0
        multi_n = output['mesh'].shape[0] // bs
        # mesh_out = output['mesh'].view(bs, multi_n, -1, 3)                        # [bs, multi_n, 10475, 3)
        mesh_out = output['mesh'].clone()                                           # [bs*multi_n, 10475, 3]
        mesh_out_ = mesh_out.view(bs, multi_n, -1, 3)                               # [bs, multi_n, 10475, 3)
        gt_mesh = gt['mesh'].clone()                                                # [bs, 10475, 3]
        gt_mesh_ = gt_mesh.unsqueeze(1)             # [bs, 1, 10475, 3]

        smplx_J_regressor = torch.from_numpy(smpl_x.J_regressor).to(mesh_out.device)        # [55, 10475]
        mesh_out_root_joint = torch.matmul(mesh_out.transpose(1, 2), smplx_J_regressor.t()).transpose(1, 2)[:, smpl_x.J_regressor_idx['pelvis'], :].reshape(bs, multi_n, 1, 3)    # [bs, multi_n, 1, 3]
        mesh_gt_root_joint = torch.matmul(gt_mesh.transpose(1, 2), smplx_J_regressor.t()).transpose(1, 2)[:, smpl_x.J_regressor_idx['pelvis'], :].unsqueeze(1).unsqueeze(1)      # [bs, 1, 1, 3]
        mesh_out_align_ = mesh_out_ - mesh_out_root_joint + mesh_gt_root_joint
        mesh_out_align = mesh_out_align_.view(bs*multi_n, -1, 3)                                                   # [bs, multi_n, 10475, 3)

        # MPVPE from all vertices
        mpvpe = torch.sqrt(torch.sum((mesh_out_align_ - gt_mesh_) ** 2, dim=3))
        mpvpe = torch.mean(mpvpe, dim=2) * 1000
        eval_result['mpvpe_all'] = mpvpe                                                                              # [bs, multi_n]

        # PA-MPVPE from all vertices
        mesh_out_align_ra_ = batch_rigid_align(mesh_out_, gt_mesh_)
        pa_mpvpe = torch.sqrt(torch.sum((mesh_out_align_ra_ - gt_mesh_) ** 2, dim=3))
        pa_mpvpe = torch.mean(pa_mpvpe, dim=2) * 1000
        eval_result['pa_mpvpe_all'] = pa_mpvpe                                                                # [bs, multi_n, 10475, 3)

        # MPJPE from body joints
        smplx_J14_regressor = torch.from_numpy(smpl_x.j14_regressor.astype(np.float32)).to(mesh_out.device)                             # [14, 10475]
        joint_gt_body = torch.matmul(gt_mesh.transpose(1, 2), smplx_J14_regressor.t()).transpose(1, 2)                 # [bs, 14, 3]
        joint_gt_body_ = joint_gt_body.unsqueeze(1)                                                                  # [bs, 1, 14, 3]
        joint_out_body = torch.matmul(mesh_out.transpose(1, 2), smplx_J14_regressor.t()).transpose(1, 2)            # [bs*multi_n, 14, 3]
        joint_out_body_ = joint_out_body.view(bs, multi_n, -1, 3)                                                    # [bs, multi_n, 14, 3]
        joint_out_body_root_align = torch.matmul(mesh_out_align.transpose(1, 2), smplx_J14_regressor.t()).transpose(1, 2)
        joint_out_body_root_align_ = joint_out_body_root_align.view(bs, multi_n, -1, 3)                             # [bs, multi_n, 14, 3]
        # mpjpe_body = np.sqrt(np.sum((joint_out_body_root_align - joint_gt_body) ** 2, 1)).mean() * 1000              # [bs, multi_n, 10475, 3)
        mpjpe_body = torch.sqrt(torch.sum((joint_out_body_root_align_ - joint_gt_body_) ** 2, dim=3))
        mpjpe_body = torch.mean(mpjpe_body, dim=2) * 1000
        eval_result['mpjpe_body'] = mpjpe_body

        # PA-MPJPE from body joints
        joint_out_body_ra_ = batch_rigid_align(joint_out_body_, joint_gt_body_)
        pa_mpjpe_body = torch.sqrt(torch.sum((joint_out_body_ra_ - joint_gt_body_) ** 2, dim=3))
        pa_mpjpe_body = torch.mean(pa_mpjpe_body, dim=2) * 1000
        eval_result['pa_mpjpe_body'] = pa_mpjpe_body

        
        score_raw_t = score_raw.view(bs, multi_n)

        mask = torch.ones(bs, device=score_raw_t.device)
        p_mppve = self.loss_func(score_raw_t.clone(), mpvpe.clone(), mask) * self.weight_mpvpe
        p_pa_mpvpe = self.loss_func(score_raw_t.clone(), pa_mpvpe.clone(), mask) * self.weight_pa_mppve
        p_mpjpe_body = self.loss_func(score_raw_t.clone(), mpjpe_body.clone(), mask) * self.weight_mpjpe
        p_pa_mpjpe_body = self.loss_func(score_raw_t.clone(), pa_mpjpe_body.clone(), mask) * self.weight_pa_mpjpe

        score_t = score.view(bs, multi_n)
        mse_mppve = self.mse_loss(score_t.clone(), error_2_score(mpvpe)) * self.weight_mpvpe_mse
        mse_pa_mpvpe = self.mse_loss(score_t.clone(), error_2_score(pa_mpvpe, max=150)) * self.weight_pa_mppve_mse
        mse_mpjpe_body = self.mse_loss(score_t.clone(), error_2_score(mpjpe_body)) * self.weight_mpjpe_mse
        mse_pa_mpjpe_body = self.mse_loss(score_t.clone(), error_2_score(pa_mpjpe_body, max=150)) * self.weight_pa_mpjpe_mse
        
        
        return p_mppve, p_pa_mpvpe, p_mpjpe_body, p_pa_mpjpe_body, mse_mppve, mse_pa_mpvpe, mse_mpjpe_body, mse_pa_mpjpe_body, eval_result
