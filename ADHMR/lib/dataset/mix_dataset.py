import bisect
import random

import torch
import torch.utils.data as data

from dataset.h36m import H36MDataset
from dataset.hp3d import HP3D
from dataset.mscoco import Mscoco
from dataset.pw3d import PW3D
from dataset.mpii import MPII
from dataset.up3d import UP3D 
from dataset.surreal import SURREAL
import logging
import numpy as np
import logging
s_mpii_2_smpl_jt = [
    6, 3, 2,
    -1, 4, 1,
    -1, 5, 0,
    -1, -1, -1,
    8, -1, -1,
    -1,
    13, 12,
    14, 11,
    15, 10,
    -1, -1
]
s_3dhp_2_smpl_jt = [
    4, -1, -1,
    -1, 19, 24,
    -1, 20, 25,
    -1, -1, -1,  
    5, -1, -1,
    -1,
    9, 14,
    10, 15,
    11, 16,
    -1, -1
]
s_coco_2_smpl_jt = [
    -1, -1, -1,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

s_smpl24_jt_num = 24


class MixDataset(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 24
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    data_domain = set([
        'target_theta',
        'target_theta_weight',
        'target_beta',
        'target_smpl_weight',
        'joints_uvd_29',
        'joints_xyz_24',
        'joints_vis_24',
        'joints_vis_29',
        'joints_xyz_17',
        'joints_vis_17',
        'trans_inv',
        'trans',
        'intrinsic_param',
        'joint_root',
        'target_twist',
        'target_twist_weight',
        'depth_factor',
        'dataset_idx',
        'sample_idx',
        'joints_2d',
        'prelate_weight'
    ])

    def __init__(self,
                 cfg,
                 train=True):
        self._train = train
        self._heatmap_size = cfg.hrnet.heatmap_size
        self._input_size = cfg.hrnet.image_size
        self.bbox_3d_shape = getattr(cfg.dataset, 'bbox_3d_shape', (2000, 2000, 2000))
        self.db0 = H36MDataset(
            cfg=cfg,
            root = cfg.dataset.set_list[0].root,
            ann_file=cfg.dataset.set_list[0].train_set,
            train=True)
        self.db1 = Mscoco(
            cfg=cfg,
            root = cfg.dataset.set_list[1].root,
            ann_file=f'person_keypoints_{cfg.dataset.set_list[1].train_set}.json',
            train=True)
        self.db2 = HP3D(
            cfg=cfg,
            root = cfg.dataset.set_list[2].root,
            ann_file=cfg.dataset.set_list[2].train_set,
            train=True)
        self.db3 = PW3D(
            cfg=cfg,
            root = cfg.dataset.set_list[3].root,
            ann_file=cfg.dataset.set_list[3].train_set,
            train=True)
        self.db4 = MPII(
            cfg=cfg,
            root = cfg.dataset.set_list[4].root,
            ann_file=cfg.dataset.set_list[4].train_set,
            train=True)
        self.db5 = UP3D(
            cfg=cfg,
            root = cfg.dataset.set_list[5].root,
            ann_file=cfg.dataset.set_list[5].train_set,
            train=True)
        self.db6 = SURREAL(
            cfg=cfg,
            root = cfg.dataset.set_list[6].root,
            ann_file=cfg.dataset.set_list[6].train_set,
            train=True)

        self._subsets = [self.db0, self.db1, self.db2, self.db3,self.db4,self.db5,self.db6]

        self._subset_size = [len(item) for item in self._subsets]
        logging.info(f"dataset length:{self._subset_size} ")
        self.use_3d = cfg.dataset.use_3d
        self.use_beta = cfg.dataset.use_beta
        self.use_29 = cfg.dataset.use_29
        self.use_twist = cfg.dataset.use_twist
        self.use_score = cfg.dataset.use_score
        self.max_db_data_num = max(self._subset_size)
        # self.tot_size = max(self._subset_size)*2
        self.tot_size = int(max(self._subset_size) * 0.3)
        if cfg.training.scorenet.train:
            self.tot_size = max(self._subset_size) 
        self.partition = cfg.dataset.partition
        logging.info(f"partition {cfg.dataset.partition}")

        self.cumulative_sizes = self.cumsum(self.partition)

        self.joint_pairs_24 = self.db0.joint_pairs_24
        self.joint_pairs_17 = self.db0.joint_pairs_17
        self.root_idx_17 = self.db0.root_idx_17
        self.root_idx_smpl = self.db0.root_idx_smpl
        

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        p = random.uniform(0, 1)

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)

        _db_len = self._subset_size[dataset_idx]

        # last batch: random sampling
        if idx >= _db_len * (self.tot_size // _db_len):
            sample_idx = random.randint(0, _db_len - 1)
        else:  # before last batch: use modular
            sample_idx = idx % _db_len


        # [DEBUG]
        # dataset_idx = 2     # 3dhp
        # sample_idx = 100000
        
        img, target, img_id, bbox, img_path = self._subsets[dataset_idx][sample_idx]
        list_smpl = [0,1,3,4,5,6]
        list_no_smpl = [2]

        if dataset_idx in list_smpl:
            target.pop('scale')
        target['dataset_idx'] = dataset_idx
        target['sample_idx'] = sample_idx
        target['prelate_weight'] = torch.ones(1).float()
        if dataset_idx not in self.use_score:
            target['prelate_weight'] = torch.zeros(1).float()

        if dataset_idx in list_no_smpl:
            # 3DHP
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)
            if dataset_idx == 2:
                label_jts_origin = label_jts_origin.reshape(28, 3)
                label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_3dhp_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                        label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()
            label_uvd_24_mask = label_uvd_29_mask[:24, :]
            target['joints_uvd_29'] = label_uvd_29
            target['joints_xyz_24'] = label_xyz_24
            target['joints_vis_24'] = label_uvd_24_mask
            target['joints_vis_29'] = label_uvd_29_mask
            target['joints_xyz_17'] = label_xyz_17
            target['joints_vis_17'] = label_xyz_17_mask
            target['target_theta'] = torch.zeros(24 * 4)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23,2)
            target['joints_2d'] = torch.zeros(29,2)
        else:
            if dataset_idx not in self.use_3d:
                target['target_theta_weight'] = torch.zeros(24 * 4)
                target['joints_vis_29'][:,2] = 0
                target['joints_vis_17'] = torch.zeros(17, 3)
            if dataset_idx not in self.use_29:
                target['joints_vis_29'][24:] = 0
            if dataset_idx not in self.use_beta:
                target['target_smpl_weight'] = torch.zeros(1)
            if dataset_idx not in self.use_twist:
                target['target_twist_weight'] = torch.zeros(23,2)
        
        assert set(target.keys()).issubset(self.data_domain), (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        return img, target, img_id, bbox
