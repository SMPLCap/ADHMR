import os
import logging
import time
import glob
from tqdm import tqdm
import json
import pickle

import numpy as np
import torch
from utils.filter_hub import *
from utils.pose_utils import *
from models.layers.smpl.lbs import rotmat_to_quat

def trans_back(joints,trans):
    a = joints[:,:,0]
    b = joints[:,:,1]
    joints_new = torch.stack([a,b,torch.ones_like(a)],dim=-1)
    trans = trans.transpose(2,1)
    joints[:,:,0:2] = torch.bmm(joints_new,trans) 
    return joints

def normalize_pose_cuda(pose, mean_and_std=None,res_w_h=None, which='zero_center',scale = None):
    batch_size = pose.shape[0]  # pose: float32 [K, 17, 2 or 3]
    dims = pose.shape[2]

    if which == 'zero_center':
        # zero-centered
        center_output = (pose - pose[:,0].reshape((batch_size, 1, -1))).float()  # torch.Size([K, 17, 2]) float
        # avoid divide by zero
        avoid_zero = np.zeros((17, 3))
        avoid_zero[0, :] = 1
        std_avoid_zero = mean_and_std['std'] + torch.from_numpy(avoid_zero).to(device='cuda', dtype=center_output.dtype)  # [17, 3]
        if dims == 2:
            pose = torch.div(center_output - mean_and_std['mean'][:, :2], std_avoid_zero[:, :2])  # torch.Size([K, 17, 2])
        else:
            pose = torch.div(center_output - mean_and_std['mean'], std_avoid_zero)  # torch.Size([K, 17, 3])
    elif which == 'scale':
        pose = (pose - pose[:,0].reshape((batch_size, 1, -1))).float()
        for idx in range(batch_size):
            res_idx = res_w_h[idx].split(' ')
            res_w, res_h = int(res_idx[0]), int(res_idx[1])
            if dims == 2:
                pose[idx, :, :] = pose[idx, :, :] / res_w  #- torch.tensor([1, res_h / res_w]).float().cuda()
            else:
                pose[idx, :, :2] = pose[idx, :, :2] / res_w  #- torch.tensor([1, res_h / res_w]).float().cuda()
                pose[idx, :, 2:] = pose[idx, :, 2:] / res_w 
    elif which == 'scale_s':
        #scale = scale.transpose(1,0)
        #print('tran',scale)
        if dims ==2:
            pose[:,:,0] = pose[:,:,0] / (scale[0].view(-1,1))
            pose[:,:,1] = pose[:,:,1] / (scale[1].view(-1,1))
        else:
            pose[:,:,0] = pose[:,:,0] / (scale[0].view(-1,1))
            pose[:,:,1] = pose[:,:,1] / (scale[1].view(-1,1))
            pose[:,:,2] = pose[:,:,2] / (scale[2].view(-1,1))
        pose = (pose - pose[:,0].reshape((batch_size, 1, -1))).float()
    elif which == 'scale_t':
        if not mean_and_std:
            mean_and_std =1
        if dims ==2:
            pose[:,:,0] = pose[:,:,0] / (scale[0].view(-1,1))- 0.5*mean_and_std
            pose[:,:,1] = pose[:,:,1] / (scale[1].view(-1,1))- 0.5*mean_and_std
        else:
            pose[:,:,0] = pose[:,:,0] / (scale[0].view(-1,1))- 0.5*mean_and_std
            pose[:,:,1] = pose[:,:,1] / (scale[1].view(-1,1))- 0.5*mean_and_std
            pose[:,:,2] = pose[:,:,2] / (scale[2].view(-1,1))
        pose = pose * 2 
    else:
        assert 0, 'only support zero_center or scale normalization'
    pose = pose.reshape((batch_size, -1))

    return pose 
def denormalize_pose_cuda(pose, mean_and_std=None,res_w_h=None, which='zero_center',scale = None,two_d = False):
    """
    pose: [N, 17*3]
    """
    batch_size = pose.shape[0]  # pose: float32 [N, 17, 3]
    if two_d:
        pose = pose.view((batch_size, -1, 2))
    else:
        pose = pose.view((batch_size, -1, 3))
    if which == 'zero_center':
        output = pose * mean_and_std['std'] + mean_and_std['mean']
        return output
    elif which == 'scale':
        for idx in range(batch_size):
            res_idx = res_w_h[idx].split(' ')
            res_w, res_h = int(res_idx[0]), int(res_idx[1])
            pose[idx, :, :2] = (pose[idx, :, :2] + torch.tensor([1, res_h / res_w]).float().cuda()) * res_w / 2
            pose[idx, :, 2:] = pose[idx, :, 2:] * res_w / 2
        return pose
    elif which == 'scale_s':
        #scale = scale.transpose(1,0)
        pose[:,:,0] = pose[:,:,0] * (scale[0].view(-1,1))
        pose[:,:,1] = pose[:,:,1] * (scale[1].view(-1,1))
        if not two_d:
            pose[:,:,2] = pose[:,:,2] * (scale[2].view(-1,1))
        return pose
    elif which == 'scale_t':
        if not mean_and_std:
            mean_and_std = 1
        pose = pose / 2
        pose[:,:,0] = (pose[:,:,0]+0.5*mean_and_std) * (scale[0].view(-1,1))
        pose[:,:,1] = (pose[:,:,1]+0.5*mean_and_std) * (scale[1].view(-1,1))
        if not two_d:
            pose[:,:,2] = pose[:,:,2] * (scale[2].view(-1,1))
        return pose

    else:
        assert 0, 'only support zero_center or scale normalization'
   
def uvd_to_cam(uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
    assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
    uvd_jts_new = uvd_jts.clone()
    assert torch.sum(torch.isnan(uvd_jts)) == 0#, ('uvd_jts', uvd_jts)
    assert torch.sum(torch.isnan(uvd_jts_new)) == 0#, ('uvd_jts_new', uvd_jts_new)
    dz = uvd_jts_new[:, :, 2]

    # transform in-bbox coordinate to image coordinate
    uv_homo_jts = torch.cat(
        (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
        dim=2)
    # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
    uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
    # transform (u,v,1) to (x,y,z)
    uv_jts_clone = uv_jts.clone()
    cam_2d_homo = torch.cat(
        (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
        dim=2)
    # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
    xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
    xyz_jts = xyz_jts.squeeze(dim=3)
    # recover absolute z : (B,K) + (B,1)
    abs_z = dz + joint_root[:, 2].unsqueeze(-1)
    # multipy absolute z : (B,K,3) * (B,K,1)
    xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

    if return_relative:
        # (B,K,3) - (B,1,3)
        xyz_jts = xyz_jts - xyz_jts[:,0].unsqueeze(1)

    #xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)
    return xyz_jts,uv_jts_clone.squeeze(-1)
    
def process_output(output_m,labels,smpl,process=False):
    batch_size = output_m['pred_joints'].size(0)
    joints = output_m['pred_joints'].clone()
    num_joints = output_m['pred_joints'].size(1)
    pred_joints,uvd_root = uvd_to_cam(joints,labels['trans_inv'].clone(), labels['intrinsic_param'].clone(), labels['joint_root'].clone(), labels['depth_factor'].clone())
    pred_joints = pred_joints - pred_joints[:,0,:].unsqueeze(1)
    pred_shape = output_m['pred_shape'].clone()
    pred_twist = output_m['pred_twist'].clone()
    pred_final = torch.zeros(batch_size,29,3).to(pred_shape.device)
    pred_final[:,:num_joints,:] = pred_joints
    output_m['pred_joints'] = pred_final.clone()*2
    output = smpl.hybrik(
            pose_skeleton=pred_final.clone().type(torch.float32) * 2,
            betas=pred_shape.type(torch.float32),
            phis=pred_twist.type(torch.float32),
            global_orient=None,
            return_verts=True,
            return_29_jts=True)
    output_m['pred_vertices'] = output.vertices.float()-output.joints_from_verts.float()[:,0].unsqueeze(1)
    pred_xyz_jts_17 = output.joints_from_verts.float() / 2
    output_m['pred_xyz_jts_17'] = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
    output_m['theta'] = rotmat_to_quat(output.rot_mats.view(-1,3,3)).view(-1,24,4)
    if process:
        n = labels['f'].shape[0]
        multi_n = batch_size // n
        pred_mesh = output_m['pred_vertices'].clone()*1000
        root_cam = torch.zeros(batch_size,1,3).to(pred_mesh.device)
        root_cam[:,:,:2] = uvd_root[:,0,:2].unsqueeze(1)
        root_cam[:,:,2] = labels['joint_root'][:, 2].clone().unsqueeze(1)
        f = labels['f'].clone().unsqueeze(1).repeat(1,multi_n,1).view(-1,2)
        c = labels['c'].clone().unsqueeze(1).repeat(1,multi_n,1).view(-1,2)
        root_img = pixel2cam_batch(root_cam,f,c)
        pred_mesh = pred_mesh + root_img
        output_m['pred_vertices'] = pred_mesh

        xyz_24 = output.joints - output.joints[:,0].unsqueeze(1)
        xyz_24 = xyz_24*1000 + root_img
        uvd_24 = cam2pixel_batch(xyz_24,f,c)[:,:,:2]
        gt_24 = uvd_root[:,:,:2].view(n,multi_n,-1,2)[:,0]
        uvd_24_compute = uvd_24.view(n,multi_n,-1,2)[:,0]

        loss = (uvd_24_compute-gt_24).square().sum(dim=-1).mean()
        output_m['loss'] = loss
        output_m['uvd_24'] = uvd_24
    return output_m

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1]), (S1.shape, S2.shape)

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    if S1.ndim == 2:
        S1_hat = compute_similarity_transform(S1.copy(), S2.copy())
    else:
        S1_hat = np.zeros_like(S1)
        for i in range(S1.shape[0]):
            S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    return S1_hat


def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def weak_cam2pixel(cam_coord, root_z, f, c):
    x = cam_coord[:, 0] / (root_z + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (root_z + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)

    avg_f = (f[0] + f[1]) / 2
    cam_param = np.array([avg_f / (root_z + 1e-8), c[0], c[1]])
    return img_coord, cam_param


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def cam2pixel_matrix(cam_coord, intrinsic_param):
    cam_coord = cam_coord.transpose(1, 0)
    cam_homogeneous_coord = np.concatenate((cam_coord, np.ones((1, cam_coord.shape[1]), dtype=np.float32)), axis=0)
    img_coord = np.dot(intrinsic_param, cam_homogeneous_coord) / (cam_coord[2, :] + 1e-8)
    img_coord = np.concatenate((img_coord[:2, :], cam_coord[2:3, :]), axis=0)
    return img_coord.transpose(1, 0)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord

def pixel2cam_batch(pixel_coord, f, c):
    cam_coord = torch.zeros_like(pixel_coord).to(pixel_coord.device)
    cam_coord[:,:,0] = (pixel_coord[:,:,0] - c[:,0].unsqueeze(1)) / f[:,0].unsqueeze(1) * pixel_coord[:,:, 2]
    cam_coord[:,:,1] = (pixel_coord[:,:,1] - c[:,1].unsqueeze(1)) / f[:,1].unsqueeze(1) * pixel_coord[:,:, 2]
    cam_coord[:,:,2] = pixel_coord[:,:, 2]
    return cam_coord

def cam2pixel_batch(cam_coord, f, c):
    img_coord = torch.zeros_like(cam_coord).to(cam_coord.device)
    img_coord[:,:,0] = cam_coord[:,:, 0] / (cam_coord[:,:, 2] + 1e-8) * f[:,0].unsqueeze(1) + c[:,0].unsqueeze(1)
    img_coord[:,:,1] = cam_coord[:,:, 1] / (cam_coord[:,:, 2] + 1e-8) * f[:,1].unsqueeze(1) + c[:,1].unsqueeze(1)
    img_coord[:,:,2] = cam_coord[:,:, 2]
    return img_coord

def pixel2cam_matrix(pixel_coord, intrinsic_param):

    x = (pixel_coord[:, 0] - intrinsic_param[0][2]) / intrinsic_param[0][0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - intrinsic_param[1][2]) / intrinsic_param[1][1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord

"""
def get_intrinsic_metrix(f, c, inv=False):
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)

    if inv:
        intrinsic_metrix[0, 0] = 1.0 / f[0]
        intrinsic_metrix[0, 2] = -c[0] / f[0]
        intrinsic_metrix[1, 1] = 1.0 / f[1]
        intrinsic_metrix[1, 2] = -c[1] / f[1]
        intrinsic_metrix[2, 2] = 1
    else:
        intrinsic_metrix[0, 0] = f[0]
        intrinsic_metrix[0, 2] = c[0]
        intrinsic_metrix[1, 1] = f[1]
        intrinsic_metrix[1, 2] = c[1]
        intrinsic_metrix[2, 2] = 1

    return intrinsic_metrix
"""

def get_intrinsic_metrix(f, c, inv=False):
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)
    intrinsic_metrix[0, 0] = f[0]
    intrinsic_metrix[0, 2] = c[0]
    intrinsic_metrix[1, 1] = f[1]
    intrinsic_metrix[1, 2] = c[1]
    intrinsic_metrix[2, 2] = 1

    if inv:
        intrinsic_metrix = np.linalg.inv(intrinsic_metrix).astype(np.float32)
    return intrinsic_metrix

def pixel2cam_test(coords, f,c):
    cam_coord = np.zeros((len(coords), 3))
    z = coords[..., 2].reshape(-1, 1)

    cam_coord[..., :2] = (coords[..., :2] - c) * z / f
    cam_coord[..., 2] = coords[..., 2]

    return cam_coord