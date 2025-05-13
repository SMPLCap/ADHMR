import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix
from nets.smpler_x import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet
from nets.scorernet import ScoreNet
from nets.loss import CoordLoss, ParamLoss, CELoss, ScoreLoss
from utils.human_models import smpl_x
from utils.transforms import rot6d_to_axis_angle, restore_bbox, axis_angle_to_rot6d, batch_rodrigues
# from utils.vis import render_mesh
from utils.preprocessing import load_img
from config import cfg
import math
import os
import open3d as o3d
import copy
import cv2
import numpy as np
from mmpose.models import build_posenet
from mmcv import Config
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps


class ScoreModel(nn.Module):
    def __init__(self, encoder, score_net):
        super(ScoreModel, self).__init__()

        # body
        self.encoder = encoder

        self.score_net = score_net

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.ce_loss = CELoss()
        self.score_loss = ScoreLoss()

        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_joint_num = len(smpl_x.pos_joint_part['rhand'])

        nums = [3, 63, 45, 45, 3]
        self.accu = []
        temp = 0
        for num in nums:
            temp += num
            self.accu.append(temp)

        self.trainable_modules = [self.score_net]

        self.backbone_trainable_modules = [self.encoder]


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(
                cfg.trainset_2d) == 0:  # prevent gradients from backpropagating to SMPLX paraemter regression module
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering
        joint_cam_wo_ra = joint_cam.clone()

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam
    
    def get_coord_proj(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, focal, princpt, rot_aug_mat, mode):
        focal = focal / 2.
        princpt = princpt / 2.
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  transl = cam_trans,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices

        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]
            joint_cam_29, _ = convert_kps(output.joints, src='smplx', dst='hybrik_29')
                
        if mode == 'train':
            # apply rot_aug matrix to joint cam in training
            joint_cam_root_trans = joint_cam[:, smpl_x.root_joint_idx][:, None]
            joint_cam = joint_cam - joint_cam_root_trans  # body root-relative
            joint_cam_root_trans_new = torch.bmm(rot_aug_mat, joint_cam_root_trans.transpose(1, 2)).transpose(1, 2)
            joint_cam = joint_cam + joint_cam_root_trans_new

            joint_cam_29_root_trans = joint_cam_29[:, 0][:, None]
            joint_cam_29 = joint_cam_29 - joint_cam_29_root_trans  # body root-relative
            joint_cam_29_root_trans_new = torch.bmm(rot_aug_mat, joint_cam_29_root_trans.transpose(1, 2)).transpose(1, 2)
            joint_cam_29 = joint_cam_29 + joint_cam_29_root_trans_new

        # project 3D coordinates to 2D space
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(
                cfg.trainset_2d) == 0:  # prevent gradients from backpropagating to SMPLX paraemter regression module
            raise NotImplementedError
        else:
            x = joint_cam[:, :, 0] / joint_cam[:, :, 2] * focal[:, 0:1] + princpt[:, 0:1]
            y = joint_cam[:, :, 1] / joint_cam[:, :, 2] * focal[:, 1: ] + princpt[:, 1: ]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        x_29 = joint_cam_29[:, :, 0] / joint_cam_29[:, :, 2] * focal[:, 0:1] + princpt[:, 0:1]
        y_29 = joint_cam_29[:, :, 1] / joint_cam_29[:, :, 2] * focal[:, 1: ] + princpt[:, 1: ]
        x_29 = x_29 / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y_29 = y_29 / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj_29 = torch.stack((x_29, y_29), 2)
        # uvd 29 depth
        joint_cam_29 = joint_cam_29 - joint_cam_29[:, 0, None, :]
        d_29 = joint_cam_29[..., 2].clone()
        u_29 = (x_29 / cfg.output_hm_shape[2] - 0.5) * 2
        v_29 = (y_29 / cfg.output_hm_shape[1] - 0.5) * 2
        joint_uvd_29 = torch.stack((u_29, v_29, d_29), 2)

        # root-relative 3D coordinates
        joint_cam_wo_ra = joint_cam.clone()
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        # mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_proj_29, joint_uvd_29, joint_cam, joint_cam_wo_ra, mesh_cam

    def generate_mesh(self, targets, mode):
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :self.accu[0]], pose[:, self.accu[0]:self.accu[1]], pose[:, self.accu[1]:self.accu[2]], pose[:, self.accu[2]:self.accu[3]], pose[:, self.accu[3]:self.accu[4]]
        # print(lhand_pose)
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        return mesh_cam, joint_proj

    def bbox_split(self, bbox):
        # bbox:[bs, 3, 3]
        lhand_bbox_center, rhand_bbox_center, face_bbox_center = \
            bbox[:, 0, :2], bbox[:, 1, :2], bbox[:, 2, :2]
        return lhand_bbox_center, rhand_bbox_center, face_bbox_center

    def forward(self, inputs, targets, meta_info, gen_output, mode):

        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)   # cropped human image body_img: 256 x 192

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # img_feat:[bs,1280,16,12]; task_token:[bs, N, c]

        # 2. Process gen_output with shape of [bs * multi_n, *]
        bs = inputs['img'].size(0)
        if mode == 'dpo':
            multi_n = gen_output['score_joints'].size(1)
            joint_uvd_29 = gen_output['score_joints'].reshape(bs * multi_n, 29, 3)
        elif mode == 'clean':
            multi_n = 1
            joint_uvd_29 = gen_output['score_joints'].reshape(bs * multi_n, 29, 3)
        else:
            if mode == 'train':
                multi_n = sum(cfg.num_hypos)
            else:
                multi_n = 1

            focal = meta_info['focal'][:, None].repeat(1, multi_n, 1).view(bs * multi_n, -1)
            princpt = meta_info['princpt'][:, None].repeat(1, multi_n, 1).view(bs * multi_n, -1)
            if mode == 'train':
                rot_aug_mat = meta_info['rot_aug_mat'][:, None].repeat(1, multi_n, 1, 1).view(bs * multi_n, 3, 3)
            else:
                rot_aug_mat = None
            smplx_shape_pred = gen_output['smplx_shape']
            cam_trans_pred = gen_output['cam_trans']                                 # [bs*multi_n, 3]
            smplx_expr_pred = gen_output['smplx_expr']
            pose_pred = gen_output['smplx_pose']
            root_pose_pred, body_pose_pred, lhand_pose_pred, rhand_pose_pred, jaw_pose_pred = \
                pose_pred[:, :self.accu[0]], pose_pred[:, self.accu[0]:self.accu[1]], pose_pred[:, self.accu[1]:self.accu[2]], \
                pose_pred[:, self.accu[2]:self.accu[3]], pose_pred[:, self.accu[3]:self.accu[4]]
            
            # get twist
            pred_joint_proj_pseudo, pred_joint_proj_29, joint_uvd_29, _, _, pred_mesh_pseudo = self.get_coord_proj(root_pose_pred, body_pose_pred, lhand_pose_pred, \
                                                rhand_pose_pred, jaw_pose_pred, smplx_shape_pred, smplx_expr_pred, cam_trans_pred, focal, princpt, rot_aug_mat, mode)
            
            root_pose_6d_pred = axis_angle_to_rot6d(root_pose_pred)
            body_pose_6d_pred = axis_angle_to_rot6d(body_pose_pred.reshape(-1, 3)).reshape(body_pose_pred.shape[0], -1)

            # 3. Process target with shape of [bs * multi_n, *]
            pose_gt = targets['smplx_pose']
            root_pose_gt, body_pose_gt, lhand_pose_gt, rhand_pose_gt, jaw_pose_gt = \
                pose_gt[:, :self.accu[0]], pose_gt[:, self.accu[0]:self.accu[1]], pose_gt[:, self.accu[1]:self.accu[2]], \
                pose_gt[:, self.accu[2]:self.accu[3]], pose_gt[:, self.accu[3]:self.accu[4]]
            gt_joint_proj_pseudo, _, _, gt_mesh_pseudo = self.get_coord(root_pose_gt, body_pose_gt, lhand_pose_gt, rhand_pose_gt, jaw_pose_gt, targets['smplx_shape'],
                                                                        targets['smplx_expr'], targets['smplx_cam_trans'], mode='test')
            
            x_body_pose = torch.cat((root_pose_6d_pred, body_pose_6d_pred), 1)            # [bs*multi_n, 6+126]
        
        # 4. Scorer
        ctx = {'global': img_feat}
        score_raw, score = self.score_net(joint_uvd_29.view(bs * multi_n, 29*3), ctx)           # bs * multi_n
        if mode == 'clean':
            out = {}
            out['score'] = score_raw
            out['img_path'] = meta_info['img_path']
            if 'ann_id' in meta_info:
                out['ann_id'] = meta_info['ann_id']
            return out
        
        if mode == 'dpo':
            score_raw_t = score_raw.reshape(bs, multi_n)
            _, sorted_indices = torch.sort(score_raw_t, dim=1, descending=True)  # descending=True 表示从大到小排序
            output_joints = targets['output_joints'].clone()            # [bs, multi_n, 29, 3]
            output_joints_sorted = output_joints.gather(1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 29, 3))
            output_twists = targets['output_twists'].clone()            # [bs, multi_n, 23, 2]
            output_twists_sorted = output_twists.gather(1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 23, 2))
            score_joints = targets['score_joints'].clone()            # [bs, multi_n, 87]
            score_joints_sorted = score_joints.gather(1, sorted_indices.unsqueeze(-1).expand(-1, -1, 87))
            score_twists = targets['score_twists'].clone()            # [bs, multi_n, 29, 3]
            score_twists_sorted = score_twists.gather(1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 23, 2))

            targets['output_joints_sorted'] = output_joints_sorted
            targets['output_twists_sorted'] = output_twists_sorted
            targets['score_joints_sorted'] = score_joints_sorted
            targets['score_twists_sorted'] = score_twists_sorted

            img_id = meta_info['img_id']

            return targets, img_id


        gt = {'mesh': gt_mesh_pseudo}
        output_final = {'mesh': pred_mesh_pseudo}

        # 5. Score Loss
        p_mppve, p_pa_mpvpe, p_mpjpe_body, p_pa_mpjpe_body, mse_mppve, mse_pa_mpvpe, mse_mpjpe_body, mse_pa_mpjpe_body, eval_result = \
            self.score_loss(score_raw, score, output_final, gt)
        if mode == 'train':
            loss = {}
            loss['p_mppve'] = p_mppve
            # reduce loss
            loss['p_pa_mpvpe'] = p_pa_mpvpe
            loss['p_mpjpe_body'] = p_mpjpe_body
            loss['p_pa_mpjpe_body'] =  p_pa_mpjpe_body


            out = {}
            out['img'] = inputs['img']
            out['score_all'] = score_raw
            out['pve_score_gt'] = eval_result['mpvpe_all']              # [bs, 1 multi_n]
            out['papve_score_gt'] = eval_result['pa_mpvpe_all']
            out['mpjpe_score_gt'] = eval_result['mpjpe_body']
            out['pa_mpjpe_score_gt'] = eval_result['pa_mpjpe_body']

            assert out['pve_score_gt'].shape[0] == out['score_all'].shape[0]

            return out


def load_model(model, state):
    state_model = model.state_dict()
    for key in state_model.keys():
        if key in state.keys() and state_model[key].shape == state[key].shape:
            state_model[key] = state[key]
            # print(key)
    model.load_state_dict(state_model)
    return model

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass

def scorehypo_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.01)

def clip_by_norm(layer, norm=1):
    if isinstance(layer, nn.Linear):
        if layer.weight.data.norm(2) > norm:
            layer.weight.data.mul_(norm / layer.weight.data.norm(2).item())


def get_scorer_model(mode):

    # body
    vit_cfg = Config.fromfile(cfg.encoder_config_file)
    vit = build_posenet(vit_cfg.model)

    scorer_cfg = Config.fromfile(cfg.scorer_config_file)

    neighbour_matrix = None
    parents = np.array([ 0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13,
                        14, 16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11])
    childrens = {0: [1, 2, 3], 1: [4], 2: [5], 3: [6], 4: [7], 5: [8], 6: [9], 7: [10], 8: [11],\
                    9: [12, 13, 14], 10: [27], 11: [28], 12: [15], 13: [16], 14: [17], 15: [24], 16: [18], \
                    17: [19], 18: [20], 19: [21], 20: [22], 21: [23], 22: [25], 23: [26], 24: [], 25: [], 26: [], 27: [], 28: []}
    neighbour_matrix = get_neighbour_matrix_from_hand(parents, childrens, num_joints=scorer_cfg.hyponet.num_joints, \
                                                      num_edges=scorer_cfg.hyponet.num_twists, knn=scorer_cfg.scorenet.knn)

    score_net = ScoreNet(scorer_cfg, neighbour_matrix)

    if mode == 'test':
        encoder = vit.backbone

    if mode == 'train':
        # body
        if not getattr(cfg, 'random_init', False):
            encoder_pretrained_model = torch.load(cfg.encoder_pretrained_model_path)['state_dict']
            vit.load_state_dict(encoder_pretrained_model, strict=False)
            print(f"Initialize encoder from {cfg.encoder_pretrained_model_path}")
        else:
            print('Random init!!!!!!!')
        
        encoder = vit.backbone
        
        if getattr(cfg, 'pretrained_model_path', None) is not None:
            ckpt_path = cfg.pretrained_model_path
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) # solve CUDA OOM error in DDP
            smplerx_weights = ckpt['network']

            from collections import OrderedDict
            encoder_weights = OrderedDict()
            for k, v in smplerx_weights.items():
                if 'module.encoder' in k:
                    new_key = k.replace('module.encoder.', '')
                    encoder_weights[new_key] = v

            encoder.load_state_dict(encoder_weights)
            print(f"Initialize encoder from SMPLer-X's encoder {cfg.pretrained_model_path}")

        if getattr(cfg, 'pretrained_scorenet_path', None):
            states_load = torch.load(cfg.pretrained_scorenet_path, map_location='cpu')
            score_net = load_model(score_net, states_load['model'])
        else:
            score_net.apply(scorehypo_init_weights)
            score_net.apply(clip_by_norm)

    model = ScoreModel(encoder, score_net)
    return model

def get_neighbour_matrix_from_hand(parents,childrens,num_joints,num_edges,knn=2):
    neighbour_matrix = np.zeros((num_joints+num_edges,num_joints+num_edges),dtype=np.float32)
    for idx in range(num_joints):
        neigbour = np.array([idx] + [parents[idx]]+childrens[idx])
        neigbour = neigbour[np.where(neigbour<num_joints)]
        neighbour_matrix[idx,neigbour] = 1
        if idx>0 and idx<=num_edges:
            neighbour_matrix[idx+num_joints-1,neigbour]=1
            neighbour_matrix[neigbour,idx+num_joints-1]=1
            neighbour_matrix[idx+num_joints-1,idx+num_joints-1]=1
    for i in range(num_joints):
        n = np.where(neighbour_matrix[i]==1)[0]
        n_edge = n[np.where(n>=num_joints)[0]]
        for edge in n_edge:
            neighbour_matrix[edge,n_edge]=1
    neighbour_matrix_raw = np.array(neighbour_matrix!=0, dtype=np.float32)
    if knn >= 2:
        neighbour_matrix = np.linalg.matrix_power(neighbour_matrix, knn)
        neighbour_matrix = np.array(neighbour_matrix!=0, dtype=np.float32)
    return neighbour_matrix,neighbour_matrix[num_joints:,num_joints:],neighbour_matrix[:num_joints,:num_joints],neighbour_matrix_raw