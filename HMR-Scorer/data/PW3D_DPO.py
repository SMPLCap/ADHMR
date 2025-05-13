'''
Score ScoreHypo's predictions on 3DPW.
'''
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import cv2
import open3d as o3d
import json
import tqdm
import copy
from tqdm import tqdm
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align, batch_rigid_align
import torch.utils.data as data


class PW3D_DPO(data.Dataset):
    def __init__(self, transform, data_split, multi_n=50):
        # super(PW3D_DPO, self).__init__(transform, data_split, multi_n=50)

        self.transform = transform

        self.img256_root_dir = '/mnt/AFS_shenwenhao/ScoreHypo/data/dpo/images/PW3D'
        self.annot_dir = '/mnt/AFS_shenwenhao/ScoreHypo/data/dpo/annotations'
        
        self.multi_n = multi_n

        self.use_cache = True               # always use cache
        self.db_dpo_path_cache = osp.join(cfg.data_dir, 'cache_dpo', f'pw3d_db_dpo_multin_{self.multi_n}.npz')
        self.dpo_id_path_cache = osp.join(cfg.data_dir, 'cache_dpo', f'pw3d_db_dpo_multin_{self.multi_n}_id.npz')

        
        if self.use_cache and osp.isfile(self.db_dpo_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.db_dpo_path_cache}')
            self.db_dpo = self.load_cache(self.db_dpo_path_cache)
            self.dpo_id_list = self.load_cache(self.dpo_id_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')

            self._dpo_ann_file = osp.join(self.annot_dir, f'new_PW3D_train_interval_1_multihypo_{multi_n}.npz')
            assert osp.exists(self._dpo_ann_file)
            self.db_dpo, self.dpo_id_list = self.load_dpo_pt()

            if self.use_cache:
                self.save_cache(self.db_dpo_path_cache, self.db_dpo)
                self.save_cache(self.dpo_id_path_cache, self.dpo_id_list)
    
    def save_cache(self, annot_path_cache, datalist):
        print(f'[{self.__class__.__name__}] Caching datalist to {annot_path_cache}...')
        Cache.save(
            annot_path_cache,
            datalist,
            data_strategy=getattr(cfg, 'data_strategy', None)
        )
    
    def load_cache(self, annot_path_cache):
        datalist = Cache(annot_path_cache)
        # assert datalist.data_strategy == getattr(cfg, 'data_strategy', None), \
        #     f'Cache data strategy {datalist.data_strategy} does not match current data strategy ' \
        #     f'{getattr(cfg, "data_strategy", None)}'
        return datalist

    def load_dpo_pt(self):
        """Load all image paths and labels from json annotation files into buffer."""
        # 'img_path': img_path[idx],
        # 'img_idx': img_ids[idx].item(),                     # absolute img_ids 
        # 'output_joints': save_pred_joints[idx],
        # 'output_twists': save_pred_twist[idx],
        # 'score_joints': save_score_joint[idx],
        # 'score_twists': save_score_twist[idx],
        # 'gt_joints': save_joints_uvd_29[idx],
        # 'gt_twists': save_w_twist[idx],
        # 'trans': save_trans[idx],
        # 'trans_inv': save_trans_inv[idx],
        
        labels = []
        dpo_id_list = []
        # with open(self._dpo_ann_file, 'r') as f:
        #     dpo_db = json.load(f)                   # TODO: slowwwwwwwwww
        dpo_db = self.load_cache(self._dpo_ann_file)
        for i, dpo_pair in enumerate(tqdm(dpo_db)):
            labels.append({
                'img_path': np.str_(dpo_pair['img_path']),
                'img_idx': np.int64(dpo_pair['img_idx']),
                'output_joints': np.array(dpo_pair['output_joints'], dtype=np.float32),
                'output_twists': np.array(dpo_pair['output_twists'], dtype=np.float32),
                'score_joints': np.array(dpo_pair['score_joints'], dtype=np.float32),
                'score_twists': np.array(dpo_pair['score_twists'], dtype=np.float32),
                'gt_joints': np.array(dpo_pair['gt_joints'], dtype=np.float32),
                'gt_twists': np.array(dpo_pair['gt_twists'], dtype=np.float32),
                'trans': np.array(dpo_pair['trans'], dtype=np.float32),
                'trans_inv': np.array(dpo_pair['trans_inv'], dtype=np.float32),
            })
            dpo_id_list.append(dpo_pair['img_idx'])
            if i == len(dpo_db) - 1:  # 跳过最后一个元素
                print(i)
                break
        return labels, dpo_id_list

    def __len__(self):
        return len(self.dpo_id_list)
    
    def __getitem__(self, idx):
        # get image id
        # img_path = self.db['img_path'][idx]
        img_id = self.dpo_id_list[idx]
        label_dpo = {}
        for k in self.db_dpo[idx].keys():
            label_dpo[k] = self.db_dpo[idx][k].copy()
        assert label_dpo['img_idx'] == img_id, "Wrong!!!"

        # img_id = self.db['img_id'][idx]
        img256_path = osp.join(self.img256_root_dir, f'{img_id:08d}.png')

        img256 = load_img(img256_path)              # (256, 256, 3))
        # transform img256 to (512, 384)
        img384 = cv2.resize(img256, (384, 384))
        top_padding = 64
        bottom_padding = 64
        img512 = cv2.copyMakeBorder(img384, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = self.transform(img512.astype(np.float32)) / 255.

        assert self.multi_n == label_dpo['score_joints'].shape[0]
        score_joints_2d = label_dpo['score_joints'].copy().reshape(self.multi_n, 29, 3)[..., :2]
        score_joints_depth = label_dpo['score_joints'].copy().reshape(self.multi_n, 29, 3)[..., 2:]


        score_joints_2d = (score_joints_2d + 1) * 127.5
        score_joints_2d = (score_joints_2d * (384 / 256)).astype(np.int32)
        score_joints_2d[..., 1] += top_padding


        score_joints_2d_norm = score_joints_2d.copy().astype(np.float32)
        score_joints_2d_norm[..., 0] = (score_joints_2d_norm[..., 0] / 191.5) - 1
        score_joints_2d_norm[..., 1] = (score_joints_2d_norm[..., 1] / 255.5) - 1

        # [multi_n, 29, 3]
        score_joints = np.concatenate((score_joints_2d_norm, score_joints_depth), axis=-1)

        inputs = {'img': img}
        targets = label_dpo
        meta_info = {'img256_path': img256_path,
                     'img_id': img_id}
        gen_output = {'score_joints': score_joints,}

        return inputs, targets, meta_info, gen_output
    

class Cache():
    """ A custom implementation for SMPLer_X pipeline
        Need to run tool/cache/fix_cache.py to fix paths
    """
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        self.load_path = load_path
        self.cache = np.load(load_path, allow_pickle=True)
        self.data_len = self.cache['data_len']
        # self.data_strategy = self.cache['data_strategy']
        assert self.data_len == len(self.cache) - 2  # data_len, data_strategy
        self.cache = None

    @classmethod
    def save(cls, save_path, data_list, data_strategy):
        assert save_path is not None, 'save_path is None'
        data_len = len(data_list)
        cache = {}
        for i, data in enumerate(data_list):
            cache[str(i)] = data
        assert len(cache) == data_len
        # update meta
        cache.update({
            'data_len': data_len,
            'data_strategy': data_strategy})

        np.savez_compressed(save_path, **cache)
        print(f'Cache saved to {save_path}.')

    # def shuffle(self):
    #     random.shuffle(self.mapping)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = np.load(self.load_path, allow_pickle=True)
        # mapped_idx = self.mapping[idx]
        # cache_data = self.cache[str(mapped_idx)]
        cache_data = self.cache[str(idx)]
        data = cache_data.item()
        return data
