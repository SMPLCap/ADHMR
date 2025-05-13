import os
import json
import os.path as osp
import numpy as np
import torch
import cv2
import random
from .humandata import HumanDataset
from utils.presets import SimpleTransform3DSMPL_DPO


class InstaVariety_DPO(HumanDataset):
    def __init__(self, cfg, transform, data_split, dpo_ann_file, dpo_root='./data/dpo'):
        super(InstaVariety_DPO, self).__init__(cfg, transform, data_split)

        self._cfg = cfg

        self.datalist = []

        pre_prc_file = 'insta_variety_neural_annot_train.npz'
        if self.data_split == 'train':
            filename = getattr(self._cfg, 'filename', pre_prc_file)
        else:
            # raise ValueError('InstaVariety test set is not support')
            filename = getattr(self._cfg, 'filename', pre_prc_file)

        self.img_dir = osp.join(self._cfg.data_dir, 'InstaVariety')
        self.annot_path = osp.join(self._cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(self._cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(self._cfg, 'use_cache', False)
        self.img_shape = (224, 224)  # (h, w)
        self.cam_param = {}

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        print(img_path)
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        self._dpo_ann_file = os.path.join(dpo_root, 'annotations', dpo_ann_file)

        self.transformation = SimpleTransform3DSMPL_DPO(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                flip = self._flip,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                scale_mult=1)
        
        # import ipdb; ipdb.set_trace()

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
            self.db_dpo, self.dpo_id_list = self.load_dpo_pt()
            print(len(self.datalist))
        else:
            self.db_dpo, self.dpo_id_list = self.load_dpo_pt()
            # import ipdb; ipdb.set_trace()
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(self._cfg, f'InstaVariety_train_sample_interval', 1),
                test_sample_interval=getattr(self._cfg, f'InstaVariety_test_sample_interval', 10))
            
            
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)


    def __len__(self):
        return len(self.dpo_id_list)

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        # get image id
        img_id = self.dpo_id_list[idx]
        img_path = self.datalist[img_id]['img_path']

        assert img_id == self.datalist[img_id]['img_id']

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.datalist[img_id].keys():
            label[k] = self.datalist[img_id][k].copy()
        label_dpo = {}
        for k in self.db_dpo[idx].keys():
            label_dpo[k] = self.db_dpo[idx][k].copy()
        # print(label_dpo['img_path'], img_path)
        assert label_dpo['img_path'] == img_path
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label, label_dpo)

        sgn = random.randint(0, 1)
        target.update({'label_sgn': sgn})
        if sgn == 1:
            target['joints_uvd_29'] = target['w_joints_uvd_29'].clone()
            target['target_twist'] = target['target_w_twist'].clone()
        else:
            target['joints_uvd_29'] = target['l_joints_uvd_29'].clone()
            target['target_twist'] = target['target_l_twist'].clone()

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox
        
    
    def load_dpo_pt(self):
        """Load all image paths and labels from json annotation files into buffer."""
        labels = []
        dpo_id_list = []
        with open(self._dpo_ann_file, 'r') as f:
            dpo_db = json.load(f)
        for _, dpo_pair in enumerate(dpo_db):
            labels.append({
                'img_path': np.str_(dpo_pair['img_path']),
                'img_idx': np.int64(dpo_pair['img_idx']),
                'l_joints': np.array(dpo_pair['l_joints']),
                'l_twist': np.array(dpo_pair['l_twist']),
                'w_joints': np.array(dpo_pair['w_joints']),
                'w_twist': np.array(dpo_pair['w_twist']),
                'trans': np.array(dpo_pair['trans']),
                'trans_inv': np.array(dpo_pair['trans_inv']),
            })
            dpo_id_list.append(dpo_pair['img_idx'])
        return labels, dpo_id_list

class InstaVariety_KTO(HumanDataset):
    def __init__(self, cfg, transform, data_split, dpo_ann_file, dpo_root='./data/dpo'):
        super(InstaVariety_KTO, self).__init__(cfg, transform, data_split)

        self._cfg = cfg

        self.datalist = []

        pre_prc_file = 'insta_variety_neural_annot_train.npz'
        if self.data_split == 'train':
            filename = getattr(self._cfg, 'filename', pre_prc_file)
        else:
            # raise ValueError('InstaVariety test set is not support')
            filename = getattr(self._cfg, 'filename', pre_prc_file)

        self.img_dir = osp.join(self._cfg.data_dir, 'InstaVariety')
        self.annot_path = osp.join(self._cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(self._cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(self._cfg, 'use_cache', False)
        self.img_shape = (224, 224)  # (h, w)
        self.cam_param = {}

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        print(img_path)
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        self._dpo_ann_file = os.path.join(dpo_root, 'annotations', dpo_ann_file)

        self.transformation = SimpleTransform3DSMPL_DPO(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                flip = self._flip,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                scale_mult=1)
        
        # import ipdb; ipdb.set_trace()

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
            self.db_dpo, self.dpo_id_list = self.load_dpo_pt()
            print(len(self.datalist))
        else:
            self.db_dpo, self.dpo_id_list = self.load_dpo_pt()
            # import ipdb; ipdb.set_trace()
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(self._cfg, f'InstaVariety_train_sample_interval', 1),
                test_sample_interval=getattr(self._cfg, f'InstaVariety_test_sample_interval', 10))
            
            
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)


    def __len__(self):
        return len(self.dpo_id_list)

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        # get image id
        img_id = self.dpo_id_list[idx]
        img_path = self.datalist[img_id]['img_path']

        assert img_id == self.datalist[img_id]['img_id']

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.datalist[img_id].keys():
            label[k] = self.datalist[img_id][k].copy()
        label_dpo = {}
        for k in self.db_dpo[idx].keys():
            label_dpo[k] = self.db_dpo[idx][k].copy()
        # print(label_dpo['img_path'], img_path)
        assert label_dpo['img_path'] == img_path
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label, label_dpo)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox
        
    
    def load_dpo_pt(self):
        """Load all image paths and labels from json annotation files into buffer."""
        labels = []
        dpo_id_list = []
        with open(self._dpo_ann_file, 'r') as f:
            dpo_db = json.load(f)
        for _, dpo_pair in enumerate(dpo_db):
            labels.append({
                'img_path': np.str_(dpo_pair['img_path']),
                'img_idx': np.int64(dpo_pair['img_idx']),
                'l_joints': np.array(dpo_pair['l_joints']),
                'l_twist': np.array(dpo_pair['l_twist']),
                'w_joints': np.array(dpo_pair['w_joints']),
                'w_twist': np.array(dpo_pair['w_twist']),
                'trans': np.array(dpo_pair['trans']),
                'trans_inv': np.array(dpo_pair['trans_inv']),
            })
            dpo_id_list.append(dpo_pair['img_idx'])
        return labels, dpo_id_list