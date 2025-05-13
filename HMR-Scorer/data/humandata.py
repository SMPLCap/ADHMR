import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D, augmentation_seg, process_db_coord_w_cam, process_human_model_output_clean
from utils.transforms import world2cam, cam2pixel, rigid_align, batch_rigid_align
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps

import tqdm
import time
import random

KPS2D_KEYS = ['keypoints2d', 'keypoints2d_smplx', 'keypoints2d_smpl', 'keypoints2d_original']
KPS3D_KEYS = ['keypoints3d_cam', 'keypoints3d', 'keypoints3d_smplx','keypoints3d_smpl' ,'keypoints3d_original'] 
# keypoints3d_cam with root-align has higher priority, followed by old version key keypoints3d
# when there is keypoints3d_smplx, use this rather than keypoints3d_original

hands_meanr = np.array([ 0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
        0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
       -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
       -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
        0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
        0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
       -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
        0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
       -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]).reshape(15, -1)
hands_meanl = np.array([ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
       -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
        0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
       -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
       -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
       -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
       -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
       -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
        0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803]).reshape(15, -1)

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
        self.data_strategy = self.cache['data_strategy']
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


class HumanDataset(torch.utils.data.Dataset):

    # same mapping for 144->137 and 190->137
    SMPLX_137_MAPPING = [
        0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63, 64, 65, 59, 58, 57, 56, 55, 37, 38, 39, 66,
        25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45,
        73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 56, 57, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143]

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # dataset information, to be filled by child class
        self.img_dir = None
        self.annot_path = None
        self.annot_path_cache = None
        self.use_cache = False
        self.save_idx = 0
        self.img_shape = None  # (h, w)
        self.cam_param = None  # {'focal_length': (fx, fy), 'princpt': (cx, cy)}
        self.use_betas_neutral = False

        self.joint_set = {
            'joint_num': smpl_x.joint_num,
            'joints_name': smpl_x.joints_name,
            'flip_pairs': smpl_x.flip_pairs}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')

    def load_cache(self, annot_path_cache):
        datalist = Cache(annot_path_cache)
        assert datalist.data_strategy == getattr(cfg, 'data_strategy', None), \
            f'Cache data strategy {datalist.data_strategy} does not match current data strategy ' \
            f'{getattr(cfg, "data_strategy", None)}'
        return datalist

    def save_cache(self, annot_path_cache, datalist):
        print(f'[{self.__class__.__name__}] Caching datalist to {self.annot_path_cache}...')
        Cache.save(
            annot_path_cache,
            datalist,
            data_strategy=getattr(cfg, 'data_strategy', None)
        )

    def load_data(self, train_sample_interval=1, test_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        num_examples = len(content['image_path'])

        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
        else:
            meta = None
            print('No meta info provided! Please give height and width manually')

        print(f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'); tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None
        
        if self.__class__.__name__ == 'HI4D':
            image_shape = None
        
        # add focal and principal point
        # import ipdb;ipdb.set_trace()
        if meta is not None and 'focal_length' in meta:
            focal_length = np.array(meta['focal_length'], dtype=np.float32)
        elif 'focal' in self.cam_param:
            focal_length = np.array(self.cam_param['focal'], dtype=np.float32)[np.newaxis].repeat(num_examples, axis=0)
        else:
            focal_length = None
        if meta is not None and 'principal_point' in meta:
            principal_point = np.array(meta['principal_point'], dtype=np.float32)
        elif 'princpt' in self.cam_param:
            principal_point = np.array(self.cam_param['princpt'], dtype=np.float32)[np.newaxis].repeat(num_examples, axis=0)
        else:
            principal_point = None

        bbox_xywh = content['bbox_xywh']

        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}

        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                        f'{content.files}')

        print('Smplx param', smplx.keys())

        if 'lhand_bbox_xywh' in content and 'rhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros_like(bbox_xywh)
            rhand_bbox_xywh = np.zeros_like(bbox_xywh)

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros_like(bbox_xywh)

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, self.SMPLX_137_MAPPING, :3] if decompressed \
                else content[kps3d_key][:, self.SMPLX_137_MAPPING, :3]
                valid_kps3d = True

                if f'{kps3d_key}_mask' in content:
                    keypoints3d_mask = content[f'{kps3d_key}_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                elif 'keypoints3d_mask' in content:
                    keypoints3d_mask = content['keypoints3d_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                break

        for kps2d_key in KPS2D_KEYS:
            if kps2d_key in content:
                keypoints2d = decompressed_kps[kps2d_key][:, self.SMPLX_137_MAPPING, :2] if decompressed \
                    else content[kps2d_key][:, self.SMPLX_137_MAPPING, :2]

                if f'{kps2d_key}_mask' in content:
                    keypoints2d_mask = content[f'{kps2d_key}_mask'][self.SMPLX_137_MAPPING]
                elif 'keypoints2d_mask' in content:
                    keypoints2d_mask = content['keypoints2d_mask'][self.SMPLX_137_MAPPING]
                break

        mask = keypoints3d_mask if valid_kps3d_mask \
                else keypoints2d_mask

        print('Done. Time: {:.2f}s'.format(time.time() - tic))

        datalist = []
        for i in tqdm.tqdm(range(int(num_examples))):
            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'test' and i % test_sample_interval != 0:
                continue
            img_path = osp.join(self.img_dir, image_path[i])
            img_shape = image_shape[i] if image_shape is not None else self.img_shape

            # Skip ARCTIC dark images
            if 'ARCTIC' in img_path:
            #    if '00001.jpg' in img_path or '00002.jpg' in img_path or '/3/' in img_path or '/7/' in img_path:
                if '00001.jpg' in img_path or '00002.jpg' in img_path:
                   continue

            bbox = bbox_xywh[i][:4]

            if hasattr(cfg, 'bbox_ratio'):
                bbox_ratio = cfg.bbox_ratio * 0.833 # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25

            if self.__class__.__name__ == 'HI4D':
                bbox_ratio = 1.25
            bbox = process_bbox(bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=bbox_ratio)
            if bbox is None: continue

            # hand/face bbox
            lhand_bbox = lhand_bbox_xywh[i]
            rhand_bbox = rhand_bbox_xywh[i]
            face_bbox = face_bbox_xywh[i]

            if lhand_bbox[-1] > 0:  # conf > 0
                lhand_bbox = lhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if lhand_bbox is not None:
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            else:
                lhand_bbox = None
            if rhand_bbox[-1] > 0:
                rhand_bbox = rhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if rhand_bbox is not None:
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
            else:
                rhand_bbox = None
            if face_bbox[-1] > 0:
                face_bbox = face_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if face_bbox is not None:
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
            else:
                face_bbox = None

            joint_img = keypoints2d[i]
            joint_valid = mask.reshape(-1, 1)
            # num_joints = joint_cam.shape[0]
            # joint_valid = np.ones((num_joints, 1))
            if valid_kps3d:
                joint_cam = keypoints3d[i]
            else:
                joint_cam = None

            smplx_param = {k: v[i] for k, v in smplx.items()}

            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3)).astype(np.float32)
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop('right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param:
                smplx_param['shape'] = smplx_param.pop('betas_neutral')

            # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['expr'] = smplx_param['expr'][:10]

            if self.__class__.__name__ == 'BEDLAM':
                smplx_param['shape'] = smplx_param['shape'][:10]
                # manually set flat_hand_mean = True
                smplx_param['lhand_pose'] -= hands_meanl
                smplx_param['rhand_pose'] -= hands_meanr

            if self.__class__.__name__ == 'HI4D':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(23, 3)


            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param['body_pose'][:21, :] # use smpl body_pose on smplx

            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx

            if smplx_param['lhand_pose'] is None:
                smplx_param['lhand_valid'] = False
            else:
                smplx_param['lhand_valid'] = True
            if smplx_param['rhand_pose'] is None:
                smplx_param['rhand_valid'] = False
            else:
                smplx_param['rhand_valid'] = True
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue

            focal = focal_length[i]
            if focal.size == 1:
                focal = focal.repeat(2)
            princpt = principal_point[i]
            if princpt.size == 1:
                princpt = princpt.repeat(2)

            seg_path = None
            if self.__class__.__name__ == 'HI4D':               # use kpts to find segmentation id
                seg_new_path = img_path.replace('/images/', '/seg/img_seg_mask/').replace('.jpg', '.png')
                id = 'all'
                seg_path = osp.join(osp.dirname(seg_new_path), id, osp.basename(seg_new_path))
                seg_img = cv2.imread(seg_path)
                assert seg_img is not None, print(seg_path)

                joint_img_vis = joint_img[smpl_x.joint_part['body']]
                n_kps = joint_img_vis.shape[0]
                # id_0_cnt, id_1_cnt, bkgd_cnt = 0, 0, 0
                id_0_color = np.array([255, 120, 28])      # id_0: blue
                id_1_color = np.array([28, 163, 255])      # id_1: orange
                bkgd_color = np.array([0, 0, 0])

                
                x_coords = np.clip(joint_img_vis[:, 0], 0, img_shape[1] - 1)
                y_coords = np.clip(joint_img_vis[:, 1], 0, img_shape[0] - 1)
                # 利用坐标一次性提取所有关键点对应的像素颜色
                try:
                    keypoint_colors = seg_img[y_coords.astype(int), x_coords.astype(int)]
                except Exception as e:
                    print(seg_path)
                    print(e)
                    exit(0)

                # 使用 NumPy 的布尔掩码来批量比较颜色
                id_0_mask = np.all(keypoint_colors == id_0_color, axis=1)
                id_1_mask = np.all(keypoint_colors == id_1_color, axis=1)
                bkgd_mask = np.all(keypoint_colors == bkgd_color, axis=1)
                # 统计每种颜色的数量
                id_0_cnt = np.sum(id_0_mask)
                id_1_cnt = np.sum(id_1_mask)
                bkgd_cnt = np.sum(bkgd_mask)

                
                id = '0' if id_0_cnt > id_1_cnt else '1'
                if bkgd_cnt == n_kps:               # if all kpts lie on the background, drop this image
                    continue
                seg_path = osp.join(osp.dirname(seg_new_path), id, osp.basename(seg_new_path))

            datalist.append({
                'img_path': img_path,
                'seg_path': seg_path,
                'img_shape': img_shape,
                'bbox': bbox,                   # body bbox [x,y,w,h] w:h=cfg.input_img_shape
                'lhand_bbox': lhand_bbox,       # lhand bbox [x0,y0,x1,y1] w:h=cfg.input_img_shape
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,         # 2D keypoints [137, 2]
                'joint_cam': joint_cam,         # 3D keypoints [137, 3]
                'joint_valid': joint_valid,     # keypoint mask[137, 1]
                'smplx_param': smplx_param,
                'focal': focal,
                'princpt': princpt,
                })

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d

        if self.data_split == 'train':
            print(f'[{self.__class__.__name__} train] original size:', int(num_examples),
                  '. Sample interval:', train_sample_interval,
                  '. Sampled size:', len(datalist))

        if (getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                getattr(cfg, 'eval_on_train', False):
            print(f'[{self.__class__.__name__}] Using [balance] strategy with datalist shuffled...')
            random.seed(2024)

            random.shuffle(datalist)

            if getattr(cfg, 'eval_on_train', False):
                return datalist[:10000]

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)

        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        seg_path = data['seg_path']

        # img
        try:
            img_path = img_path.replace('.jpg', '.jpeg') if not osp.exists(img_path) else img_path
            # print(img_path)
            img = load_img(img_path)
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            print(img_path)
            return self[idx+1]
        seg_img = None
        if seg_path is not None:
            seg_img = load_img(seg_path)
            img, seg_img, img2bb_trans, bb2img_trans, rot, do_flip, focal_scale = augmentation_seg(img, seg_img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.
            seg_img = self.transform(seg_img.astype(np.float32)) / 255.
        else:
            img, img2bb_trans, bb2img_trans, rot, do_flip, focal_scale = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.
            seg_img = torch.zeros_like(img)
        
        if self.data_split == 'train':
            # h36m gt
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord: 
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth
            
            smplx_param = data['smplx_param']
            smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None
            focal = np.array(data['focal']) if 'focal' in data else None
            princpt = np.array(data['princpt']) if 'princpt' in data else None
            if princpt.size == 1:
                princpt = princpt.repeat(2)
            if focal is not None and princpt is not None:
                focal = focal * focal_scale
                joint_img_aug, princpt, cam_trans, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc, rot_aug_mat = process_db_coord_w_cam(
                    joint_img, joint_cam, princpt, smplx_cam_trans, data['joint_valid'], do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            else:
                joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                    joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            
            # smplx coordinates and parameters
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img,  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata
            )

            # TODO temp fix keypoints3d for renbody
            if 'RenBody' in self.__class__.__name__:
                joint_cam_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['face'], :] = joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                                + joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative

            # change smplx_shape if use_betas_neutral
            # processing follows that in process_human_model_output
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(1, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(-1)
                
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
            if not (smplx_shape == 0).all():
                smplx_shape_valid = True
            else: 
                smplx_shape_valid = False

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
            face_bbox_size = face_bbox[1] - face_bbox[0]

            datatype = 0.    # 0: full-body; 1: hand dataset
            if self.__class__.__name__ in ['FreiHand', 'InterHand', 'BlurHand', 'HanCo']:
                datatype = 1.

            inputs = {'img': img,
                      'seg_img': seg_img}
            targets = {'joint_img': joint_img_aug, # keypoints2d with depth, in hm space [137, 3]
                       'smplx_joint_img': joint_img_aug, #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
                       'joint_cam': joint_cam_wo_ra, # joint_cam actually not used in any loss, # raw kps3d probably without ra
                       'smplx_joint_cam': smplx_joint_cam if dummy_cord else joint_cam_ra, # kps3d with body, face, hand root aligned
                       'smplx_pose': smplx_pose,
                       'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr,
                       'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
                       'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
                       'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size,
                       'smplx_cam_trans' : cam_trans.astype(np.float32),
                       }
            meta_info = {'joint_valid': joint_valid,
                         'joint_trunc': joint_trunc,
                         'smplx_joint_valid': smplx_joint_valid if dummy_cord else joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc if dummy_cord else joint_trunc,
                         'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid),
                         'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False) if dummy_cord else float(True), 
                         'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid,
                         'datatype': datatype,
                         'focal': focal,
                         'princpt': princpt,
                         'rot_aug_mat': rot_aug_mat,}

            
            if self.__class__.__name__  == 'SHAPY':
                meta_info['img_path'] = img_path
            
            return inputs, targets, meta_info

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__  == 'SHAPY':
            inputs = {'img': img}
            if cfg.shapy_eval_split == 'val':
                targets = {'smplx_shape': smplx_shape}
            else:
                targets = {}
            meta_info = {'img_path': img_path}
            return inputs, targets, meta_info

        else:
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord:
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth

            smplx_param = data['smplx_param']
            smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None
            focal = np.array(data['focal']) if 'focal' in data else None
            princpt = np.array(data['princpt']) if 'princpt' in data else None
            if princpt.size == 1:
                princpt = princpt.repeat(2)
            if focal is not None and princpt is not None:
                focal = focal * focal_scale
                joint_img, princpt, cam_trans, joint_cam, joint_cam_ra, joint_valid, joint_trunc, rot_aug_mat = process_db_coord_w_cam(
                    joint_img, joint_cam, princpt, smplx_cam_trans, data['joint_valid'], do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            else:
                joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                    joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)

            # smplx coordinates and parameters
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img
                )  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata

            inputs = {'img': img}
            targets = {'smplx_pose': smplx_pose,
                       'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr,
                       'smplx_cam_trans' : cam_trans,
                       }
            meta_info = {'img_path': img_path,
                         'bb2img_trans': bb2img_trans,
                         'gt_smplx_transl':cam_trans,
                         'focal': focal,
                         'princpt': princpt,
                         }

            return inputs, targets, meta_info

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0])
            xmax = np.max(bbox[:, 0])
            ymin = np.min(bbox[:, 1])
            ymax = np.max(bbox[:, 1])
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def evaluate(self, outs, cur_sample_idx=None):
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                       'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                       'pa_mpjpe_body': [], 'pa_mpjpe_l_hand': [], 'pa_mpjpe_r_hand': [], 'pa_mpjpe_hand': []}

        if getattr(cfg, 'vis', False):
            import csv
            csv_file = f'{cfg.vis_dir}/{cfg.testset}_smplx_error.csv'
            file = open(csv_file, 'a', newline='')
            writer = csv.writer(file)

        for n in range(sample_num):
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_pseudo_gt']
            mesh_out = out['smplx_mesh_cam']

            # MPVPE from all vertices
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None,
                                             :]
            mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['mpvpe_all'].append(mpvpe_all)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            pa_mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]
            eval_result['mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                                  None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                             smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
            joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
            eval_result['pa_mpjpe_l_hand'].append(np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpjpe_r_hand'].append(np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpjpe_hand'].append((np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            if getattr(cfg, 'vis', False):
                img_path = out['img_path']
                rel_img_path = img_path.split('..')[-1]
                smplx_pred = {}
                smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3)
                smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3)
                smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3)
                smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3)
                smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3)
                smplx_pred['leye_pose'] = np.zeros((1, 3))
                smplx_pred['reye_pose'] = np.zeros((1, 3))
                smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10)
                smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10)
                smplx_pred['transl'] =  out['gt_smplx_transl'].reshape(-1,3)
                smplx_pred['img_path'] = rel_img_path
                
                npz_path = os.path.join(cfg.vis_dir, f'{self.save_idx}.npz')
                np.savez(npz_path, **smplx_pred)

                # save img path and error
                new_line = [self.save_idx, rel_img_path, mpvpe_all, pa_mpvpe_all]
                # Append the new line to the CSV file
                writer.writerow(new_line)
                self.save_idx += 1

        if getattr(cfg, 'vis', False):
            file.close()

        return eval_result
            
    def print_eval_result(self, eval_result):
        print(f'======{cfg.testset}======')
        print(f'{cfg.vis_dir}')
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        print('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        print('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print()

        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        print('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))

        print()
        print(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")
        print()


        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'{cfg.testset} dataset \n')
        f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        f.write('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        f.write('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        f.write('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        f.write('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))
        f.write(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")

        if getattr(cfg, 'eval_on_train', False):
            import csv
            csv_file = f'{cfg.root_dir}/output/{cfg.testset}_eval_on_train.csv'
            exp_id = cfg.exp_name.split('_')[1]
            new_line = [exp_id,np.mean(eval_result['pa_mpvpe_all']),np.mean(eval_result['pa_mpvpe_l_hand']),np.mean(eval_result['pa_mpvpe_r_hand']),np.mean(eval_result['pa_mpvpe_hand']),np.mean(eval_result['pa_mpvpe_face']),
                        np.mean(eval_result['mpvpe_all']),np.mean(eval_result['mpvpe_l_hand']),np.mean(eval_result['mpvpe_r_hand']),np.mean(eval_result['mpvpe_hand']),np.mean(eval_result['mpvpe_face']),
                        np.mean(eval_result['pa_mpjpe_body']),np.mean(eval_result['pa_mpjpe_l_hand']),np.mean(eval_result['pa_mpjpe_r_hand']),np.mean(eval_result['pa_mpjpe_hand'])]

            # Append the new line to the CSV file
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_line)

    def decompress_keypoints(self, humandata) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be inserted to the right places and f'{key}_mask'
        will be unlocked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert bool(humandata['__keypoints_compressed__']) is True
        key_pairs = []
        for key in humandata.files:
            if key not in KPS2D_KEYS + KPS3D_KEYS:
                continue
            mask_key = f'{key}_mask'
            if mask_key in humandata.files:
                print(f'Decompress {key}...')
                key_pairs.append([key, mask_key])
        decompressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            mask_array = np.asarray(humandata[mask_key])
            compressed_kpt = humandata[kpt_key]
            kpt_array = \
                self.add_zero_pad(compressed_kpt, mask_array)
            decompressed_dict[kpt_key] = kpt_array
        del humandata
        return decompressed_dict

    def add_zero_pad(self, compressed_array: np.ndarray,
                         mask_array: np.ndarray) -> np.ndarray:
        """Pad zeros to a compressed keypoints array.

        Args:
            compressed_array (np.ndarray):
                A compressed keypoints array.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A keypoints array in full-size.
        """
        assert mask_array.sum() == compressed_array.shape[1]
        data_len, _, dim = compressed_array.shape
        mask_len = mask_array.shape[0]
        ret_value = np.zeros(
            shape=[data_len, mask_len, dim], dtype=compressed_array.dtype)
        valid_mask_index = np.where(mask_array == 1)[0]
        ret_value[:, valid_mask_index, :] = compressed_array
        return ret_value

    def construct_scorer_testset(self, multi_n=1, use_cache=True):
        save_pve_list, save_mpjpe_list, save_papve_list, save_pa_mpjpe_list = [], [], [], []
        pairwise_save_annot_path = osp.join(cfg.data_dir, 'cache_scorer_eval', 
                f'pairwise_{self.__class__.__name__}_{self.data_split}_interval_{self.test_sample_interval}_multin_{multi_n}.npz')
        pointwise_save_annot_path = osp.join(cfg.data_dir, 'cache_scorer_eval', 
                f'pointwise_{self.__class__.__name__}_{self.data_split}_interval_{self.test_sample_interval}_multin_{multi_n}.npz')
        self.pairwise_datalist = []
        self.pointwise_datalist = []
        for idx in tqdm.tqdm(range(len(self.datalist))):
            data = copy.deepcopy(self.datalist[idx])
            pairwise_data = copy.deepcopy(data)
            pointwise_data = copy.deepcopy(data)
            human_model_param = data['smplx_param']

            root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
                                             human_model_param['shape'], human_model_param['trans']
            
            if 'lhand_pose' in human_model_param and human_model_param['lhand_valid']:
                lhand_pose = human_model_param['lhand_pose']
            else:
                lhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
            if 'rhand_pose' in human_model_param and human_model_param['rhand_valid']:
                rhand_pose = human_model_param['rhand_pose']
            else:
                rhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
            if 'jaw_pose' in human_model_param and 'expr' in human_model_param and human_model_param['face_valid']:
                jaw_pose = human_model_param['jaw_pose']
                expr = human_model_param['expr']
            else:
                jaw_pose = np.zeros((3), dtype=np.float32)
                expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            if 'gender' in human_model_param:
                gender = human_model_param['gender']
            else:
                gender = 'neutral'

            root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
            body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
            lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
            rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
            jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
            shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
            expr = torch.FloatTensor(expr).view(1, -1)  # SMPLX expression parameter
            trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

            # apply camera extrinsic (rotation)
            # merge root pose and camera rotation
            if 'R' in self.cam_param:
                R = np.array(self.cam_param['R'], dtype=np.float32).reshape(3, 3)
                root_pose = root_pose.numpy()
                root_pose, _ = cv2.Rodrigues(root_pose)
                root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
                root_pose = torch.from_numpy(root_pose).view(1, 3)

            # gt mesh
            zero_pose = torch.zeros((1, 3)).float()  # eye poses
            with torch.no_grad():
                gt_output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                            transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                            right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                            leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
            gt_mesh = gt_output.vertices                    # [1, 10475, 3]

            
            # ADD random Noise to GT smplx params
            noise_var_shape = getattr(cfg, 'noise_var_shape', 0.05)
            noise_var_cam = getattr(cfg, 'noise_var_cam', 0.)
            noise_var_expr = getattr(cfg, 'noise_var_expr', 0.01)
            noise_var_body_pose = getattr(cfg, 'noise_var_pose', 0.05)
            noise_var_root_pose = getattr(cfg, 'noise_var_root_pose', 0.02)

            root_pose = root_pose[None].repeat(multi_n, 1, 1).flatten(1, 2)
            root_pose = root_pose + torch.randn_like(root_pose) * noise_var_root_pose
            body_pose = body_pose[None].repeat(multi_n, 1, 1).flatten(1, 2)
            body_pose = body_pose + torch.randn_like(body_pose) * noise_var_body_pose
            lhand_pose = lhand_pose[None].repeat(multi_n, 1, 1).flatten(1, 2)
            lhand_pose = lhand_pose + torch.randn_like(lhand_pose) * noise_var_body_pose
            rhand_pose = rhand_pose[None].repeat(multi_n, 1, 1).flatten(1, 2)
            rhand_pose = rhand_pose + torch.randn_like(rhand_pose) * noise_var_body_pose
            jaw_pose = jaw_pose[None].repeat(multi_n, 1, 1).flatten(1, 2)
            shape = shape[None].repeat(multi_n, 1, 1).flatten(1, 2)
            shape = shape + torch.randn_like(shape) * noise_var_shape
            expr = expr[None].repeat(multi_n, 1, 1).flatten(1, 2)
            expr = expr + torch.randn_like(expr) * noise_var_expr
            trans = trans[None].repeat(multi_n, 1, 1).flatten(1, 2)
            trans = trans + torch.randn_like(trans) * noise_var_cam

            zero_pose = torch.zeros((multi_n, 3)).float()  # eye poses
            with torch.no_grad():
                pred_output = smpl_x.layer[gender](betas=shape, body_pose=body_pose, global_orient=root_pose,
                                            transl=trans, left_hand_pose=lhand_pose,
                                            right_hand_pose=rhand_pose, jaw_pose=jaw_pose,
                                            leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
            pred_mesh = pred_output.vertices                    # [multi_n, 10475, 3]

            eval_result = {}
            bs = gt_mesh.shape[0]
            # assert pred_mesh.shape[0] % bs == 0
            assert multi_n == pred_mesh.shape[0] // bs
            # mesh_out = output['mesh'].view(bs, multi_n, -1, 3)                        # [bs, multi_n, 10475, 3)
            mesh_out = pred_mesh.clone()                                           # [bs*multi_n, 10475, 3]
            mesh_out_ = mesh_out.view(bs, multi_n, -1, 3)                               # [bs, multi_n, 10475, 3)
            gt_mesh = gt_mesh.clone()                                                # [bs, 10475, 3]
            gt_mesh_ = gt_mesh.unsqueeze(1)                                         # [bs, 1, 10475, 3]

            smplx_J_regressor = torch.from_numpy(smpl_x.J_regressor).to(mesh_out.device)        # [55, 10475]
            mesh_out_root_joint = torch.matmul(mesh_out.transpose(1, 2), smplx_J_regressor.t()).transpose(1, 2)[:, smpl_x.J_regressor_idx['pelvis'], :].reshape(bs, multi_n, 1, 3)    # [bs, multi_n, 1, 3]
            mesh_gt_root_joint = torch.matmul(gt_mesh.transpose(1, 2), smplx_J_regressor.t()).transpose(1, 2)[:, smpl_x.J_regressor_idx['pelvis'], :].unsqueeze(1).unsqueeze(1)      # [bs, 1, 1, 3]
            mesh_out_align_ = mesh_out_ - mesh_out_root_joint + mesh_gt_root_joint
            mesh_out_align = mesh_out_align_.view(bs*multi_n, -1, 3)                                                   # [bs, multi_n, 10475, 3)

            # MPVPE from all vertices
            mpvpe = torch.sqrt(torch.sum((mesh_out_align_ - gt_mesh_) ** 2, dim=3))
            mpvpe = torch.mean(mpvpe, dim=2) * 1000
            eval_result['mpvpe_all'] = mpvpe[0].numpy().astype(np.float32)
            
            # PA-MPVPE from all vertices
            mesh_out_align_ra_ = batch_rigid_align(mesh_out_, gt_mesh_)
            pa_mpvpe = torch.sqrt(torch.sum((mesh_out_align_ra_ - gt_mesh_) ** 2, dim=3))
            pa_mpvpe = torch.mean(pa_mpvpe, dim=2) * 1000
            eval_result['pa_mpvpe_all'] = pa_mpvpe[0].numpy().astype(np.float32)                                                                # [bs, multi_n, 10475, 3)

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
            eval_result['mpjpe_body'] = mpjpe_body[0].numpy().astype(np.float32)

            # PA-MPJPE from body joints
            joint_out_body_ra_ = batch_rigid_align(joint_out_body_, joint_gt_body_)
            pa_mpjpe_body = torch.sqrt(torch.sum((joint_out_body_ra_ - joint_gt_body_) ** 2, dim=3))
            pa_mpjpe_body = torch.mean(pa_mpjpe_body, dim=2) * 1000
            eval_result['pa_mpjpe_body'] = pa_mpjpe_body[0].numpy().astype(np.float32)

            save_pve_list.append(eval_result['mpvpe_all'])
            save_mpjpe_list.append(eval_result['mpjpe_body'])
            save_papve_list.append(eval_result['pa_mpvpe_all'])
            save_pa_mpjpe_list.append(eval_result['pa_mpjpe_body'])

            # Save pair-wise data item
            smplx_param_preds = {
                'body_pose': body_pose.numpy().reshape(multi_n, 21, 3), 
                'jaw_pose': jaw_pose.numpy(), 
                'leye_pose': zero_pose.numpy(), 
                'reye_pose': zero_pose.numpy(),
                'root_pose': root_pose.numpy(),
                'shape': shape.numpy(),
                'trans': trans.numpy(),
                'lhand_pose': lhand_pose.numpy().reshape(multi_n, 15, 3),
                'rhand_pose': rhand_pose.numpy().reshape(multi_n, 15, 3),
                'expr': expr.numpy(),
                'lhand_valid': human_model_param['lhand_valid'],
                'rhand_valid': human_model_param['rhand_valid'],
                'face_valid': human_model_param['face_valid']
            }
            pairwise_data['smplx_param_preds'] = smplx_param_preds
            pairwise_data['multi_n'] = multi_n
            pairwise_data.update(eval_result)
            self.pairwise_datalist.append(copy.deepcopy(pairwise_data))

            # Save point-wise data item
            pointwise_data['multi_n'] = multi_n
            for point_idx in range(multi_n):
                smplx_param_pred = {
                    'body_pose': body_pose.numpy()[point_idx].reshape(21, 3), 
                    'jaw_pose': jaw_pose.numpy()[point_idx].reshape(3,),
                    'leye_pose': zero_pose.numpy()[point_idx].reshape(3,),
                    'reye_pose': zero_pose.numpy()[point_idx].reshape(3,),
                    'root_pose': root_pose.numpy()[point_idx].reshape(3,),
                    'shape': shape.numpy()[point_idx].reshape(10),
                    'trans': trans.numpy()[point_idx].reshape(3,),
                    'lhand_pose': lhand_pose.numpy()[point_idx].reshape(15, 3),
                    'rhand_pose': rhand_pose.numpy()[point_idx].reshape(15, 3),
                    'expr': expr.numpy()[point_idx].reshape(10,),
                    'lhand_valid': human_model_param['lhand_valid'],
                    'rhand_valid': human_model_param['rhand_valid'],
                    'face_valid': human_model_param['face_valid']
                }
                pointwise_data['smplx_param_pred'] = smplx_param_pred
                point_eval_result = {kk: vv[point_idx] for kk, vv in eval_result.items()}
                pointwise_data.update(point_eval_result)
                self.pointwise_datalist.append(copy.deepcopy(pointwise_data))


        if use_cache:
            assert len(self.pointwise_datalist) // len(self.pairwise_datalist) == multi_n, "Strange"
            self.save_cache(pairwise_save_annot_path, self.pairwise_datalist)
            self.save_cache(pointwise_save_annot_path, self.pointwise_datalist)


class HumanDataset_Clean(HumanDataset):

    # same mapping for 144->137 and 190->137
    SMPLX_137_MAPPING = [
        0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63, 64, 65, 59, 58, 57, 56, 55, 37, 38, 39, 66,
        25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45,
        73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 56, 57, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143]

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # dataset information, to be filled by child class
        self.img_dir = None
        self.annot_path = None
        self.annot_path_cache = None
        self.use_cache = False
        self.save_idx = 0
        self.img_shape = None  # (h, w)
        self.cam_param = None  # {'focal_length': (fx, fy), 'princpt': (cx, cy)}
        self.use_betas_neutral = False

        self.joint_set = {
            'joint_num': smpl_x.joint_num,
            'joints_name': smpl_x.joints_name,
            'flip_pairs': smpl_x.flip_pairs}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')


    def load_data(self, train_sample_interval=1, test_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        num_examples = len(content['image_path'])

        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
        else:
            meta = None
            print('No meta info provided! Please give height and width manually')

        print(f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'); tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None
        
        if self.__class__.__name__ == 'HI4D':
            image_shape = None
        
        # add focal and principal point
        if meta is not None and 'focal_length' in meta:
            focal_length = np.array(meta['focal_length'], dtype=np.float32)
        elif 'focal' in self.cam_param:
            focal_length = np.array(self.cam_param['focal'], dtype=np.float32)[np.newaxis].repeat(num_examples, axis=0)
        else:
            focal_length = None
        if meta is not None and 'principal_point' in meta:
            principal_point = np.array(meta['principal_point'], dtype=np.float32)
        elif 'princpt' in self.cam_param:
            principal_point = np.array(self.cam_param['princpt'], dtype=np.float32)[np.newaxis].repeat(num_examples, axis=0)
        else:
            principal_point = None

        bbox_xywh = content['bbox_xywh']

        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}

        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                        f'{content.files}')

        print('Smplx param', smplx.keys())

        if 'lhand_bbox_xywh' in content and 'rhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros_like(bbox_xywh)
            rhand_bbox_xywh = np.zeros_like(bbox_xywh)

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros_like(bbox_xywh)

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, :144, :3] if decompressed \
                else content[kps3d_key][:, :144, :3]
                valid_kps3d = True

                if f'{kps3d_key}_mask' in content:
                    keypoints3d_mask = content[f'{kps3d_key}_mask'][:144]
                    valid_kps3d_mask = True
                elif 'keypoints3d_mask' in content:
                    keypoints3d_mask = content['keypoints3d_mask'][:144]
                    valid_kps3d_mask = True
                break

        for kps2d_key in KPS2D_KEYS:
            if kps2d_key in content:
                keypoints2d = decompressed_kps[kps2d_key][:, :144, :2] if decompressed \
                    else content[kps2d_key][:, :144, :2]

                if f'{kps2d_key}_mask' in content:
                    keypoints2d_mask = content[f'{kps2d_key}_mask'][:144]
                elif 'keypoints2d_mask' in content:
                    keypoints2d_mask = content['keypoints2d_mask'][:144]
                break

        mask = keypoints3d_mask if valid_kps3d_mask \
                else keypoints2d_mask

        print('Done. Time: {:.2f}s'.format(time.time() - tic))
                
        datalist = []
        for i in tqdm.tqdm(range(int(num_examples))):
            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'test' and i % test_sample_interval != 0:
                continue
            img_path = osp.join(self.img_dir, image_path[i])
            img_shape = image_shape[i] if image_shape is not None else self.img_shape

            # Skip ARCTIC dark images
            if 'ARCTIC' in img_path:
            #    if '00001.jpg' in img_path or '00002.jpg' in img_path or '/3/' in img_path or '/7/' in img_path:
                if '00001.jpg' in img_path or '00002.jpg' in img_path:
                   continue

            bbox = bbox_xywh[i][:4]

            if hasattr(cfg, 'bbox_ratio'):
                bbox_ratio = cfg.bbox_ratio * 0.833 # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25

            if self.__class__.__name__ == 'HI4D':
                bbox_ratio = 1.25
            bbox = process_bbox(bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=bbox_ratio)
            if bbox is None: continue

            # hand/face bbox
            lhand_bbox = lhand_bbox_xywh[i]
            rhand_bbox = rhand_bbox_xywh[i]
            face_bbox = face_bbox_xywh[i]

            if lhand_bbox[-1] > 0:  # conf > 0
                lhand_bbox = lhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if lhand_bbox is not None:
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            else:
                lhand_bbox = None
            if rhand_bbox[-1] > 0:
                rhand_bbox = rhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if rhand_bbox is not None:
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
            else:
                rhand_bbox = None
            if face_bbox[-1] > 0:
                face_bbox = face_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if face_bbox is not None:
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
            else:
                face_bbox = None

            joint_img = keypoints2d[i]
            joint_valid = mask.reshape(-1, 1)
            # num_joints = joint_cam.shape[0]
            # joint_valid = np.ones((num_joints, 1))
            if valid_kps3d:
                joint_cam = keypoints3d[i]
            else:
                joint_cam = None

            smplx_param = {k: v[i] for k, v in smplx.items()}

            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3)).astype(np.float32)
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop('right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param:
                smplx_param['shape'] = smplx_param.pop('betas_neutral')

            # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['expr'] = smplx_param['expr'][:10]

            if self.__class__.__name__ == 'BEDLAM':
                smplx_param['shape'] = smplx_param['shape'][:10]
                # manually set flat_hand_mean = True
                smplx_param['lhand_pose'] -= hands_meanl
                smplx_param['rhand_pose'] -= hands_meanr

            if self.__class__.__name__ == 'ARCTIC':
                smplx_param['lhand_pose'] -= hands_meanl
                smplx_param['rhand_pose'] -= hands_meanr

            
            if self.__class__.__name__ == 'HI4D':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(23, 3)


            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param['body_pose'][:21, :] # use smpl body_pose on smplx

            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx

            if smplx_param['lhand_pose'] is None:
                smplx_param['lhand_valid'] = False
            else:
                smplx_param['lhand_valid'] = True
            if smplx_param['rhand_pose'] is None:
                smplx_param['rhand_valid'] = False
            else:
                smplx_param['rhand_valid'] = True
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue

            focal = focal_length[i]
            if focal.size == 1:
                focal = focal.repeat(2)
            princpt = principal_point[i]
            if princpt.size == 1:
                princpt = princpt.repeat(2)


            seg_path = None
            if self.__class__.__name__ == 'HI4D':               # use kpts to find segmentation id
                seg_new_path = img_path.replace('/images/', '/seg/img_seg_mask/').replace('.jpg', '.png')
                id = 'all'
                seg_path = osp.join(osp.dirname(seg_new_path), id, osp.basename(seg_new_path))
                seg_img = cv2.imread(seg_path)

                joint_img_vis = joint_img[smpl_x.joint_part['body']]
                n_kps = joint_img_vis.shape[0]
                # id_0_cnt, id_1_cnt, bkgd_cnt = 0, 0, 0
                id_0_color = np.array([255, 120, 28])      # id_0: blue
                id_1_color = np.array([28, 163, 255])      # id_1: orange
                bkgd_color = np.array([0, 0, 0])

                
                x_coords = np.clip(joint_img_vis[:, 0], 0, img_shape[1] - 1)
                y_coords = np.clip(joint_img_vis[:, 1], 0, img_shape[0] - 1)
                # 利用坐标一次性提取所有关键点对应的像素颜色
                try:
                    keypoint_colors = seg_img[y_coords.astype(int), x_coords.astype(int)]
                except Exception as e:
                    print(seg_path)
                    print(e)
                    exit(0)

                id_0_mask = np.all(keypoint_colors == id_0_color, axis=1)
                id_1_mask = np.all(keypoint_colors == id_1_color, axis=1)
                bkgd_mask = np.all(keypoint_colors == bkgd_color, axis=1)

                id_0_cnt = np.sum(id_0_mask)
                id_1_cnt = np.sum(id_1_mask)
                bkgd_cnt = np.sum(bkgd_mask)

                
                id = '0' if id_0_cnt > id_1_cnt else '1'
                if bkgd_cnt == n_kps:               # if all kpts lie on the background, drop this image
                    continue
                seg_path = osp.join(osp.dirname(seg_new_path), id, osp.basename(seg_new_path))

            datalist.append({
                'ann_id': i,
                'img_path': img_path,
                'seg_path': seg_path,
                'img_shape': img_shape,
                'bbox': bbox,                   # body bbox [x,y,w,h] w:h=cfg.input_img_shape
                'lhand_bbox': lhand_bbox,       # lhand bbox [x0,y0,x1,y1] w:h=cfg.input_img_shape
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,         # 2D keypoints [144, 2]
                'joint_cam': joint_cam,         # 3D keypoints [144, 3]
                'joint_valid': joint_valid,     # keypoint mask[144, 1]
                'smplx_param': smplx_param,     # 修改后的smplx标注
                'focal': focal,
                'princpt': princpt,
                # 'smplx': smplx,
                })

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d

        if self.data_split == 'train':
            print(f'[{self.__class__.__name__} train] original size:', int(num_examples),
                  '. Sample interval:', train_sample_interval,
                  '. Sampled size:', len(datalist))

        if (getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                getattr(cfg, 'eval_on_train', False):
            print(f'[{self.__class__.__name__}] Using [balance] strategy with datalist NOT shuffled...')

            if getattr(cfg, 'eval_on_train', False):
                return datalist[:10000]

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)

        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        seg_path = data['seg_path']

        # img
        img = load_img(img_path)
        seg_img = None
        if seg_path is not None:
            seg_img = load_img(seg_path)
            img, seg_img, img2bb_trans, bb2img_trans, rot, do_flip, focal_scale = augmentation_seg(img, seg_img, bbox, 'test')
            img = self.transform(img.astype(np.float32)) / 255.
            seg_img = self.transform(seg_img.astype(np.float32)) / 255.
        else:
            img, img2bb_trans, bb2img_trans, rot, do_flip, focal_scale = augmentation(img, bbox, 'test')
            img = self.transform(img.astype(np.float32)) / 255.
            seg_img = torch.zeros_like(img)

        if self.data_split == 'train':
            # h36m gt
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord: 
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth
            
            smplx_param = data['smplx_param']
            smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None
            focal = np.array(data['focal']) if 'focal' in data else None
            princpt = np.array(data['princpt']) if 'princpt' in data else None
            if princpt.size == 1:
                princpt = princpt.repeat(2)
            if focal is not None and princpt is not None:
                focal = focal * focal_scale
                joint_img_aug, princpt, cam_trans, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc, rot_aug_mat = process_db_coord_w_cam(
                    joint_img, joint_cam, princpt, smplx_cam_trans, data['joint_valid'], do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            else:
                joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                    joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                    self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            
            # smplx coordinates and parameters
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output_clean(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img,  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata
            )

            # TODO temp fix keypoints3d for renbody
            if 'RenBody' in self.__class__.__name__:
                joint_cam_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['face'], :] = joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                                + joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative


            
            if self.__class__.__name__  == 'SHAPY':
                meta_info['img_path'] = img_path
            
            smplx_joint_img_29, _ = convert_kps(smplx_joint_img[np.newaxis], src='smplx', dst='hybrik_29')
            smplx_joint_img_29 = smplx_joint_img_29[0]
            x_29_norm = (smplx_joint_img_29[:, 0, np.newaxis] / cfg.output_hm_shape[2] - 0.5) * 2.
            y_29_norm = (smplx_joint_img_29[:, 1, np.newaxis] / cfg.output_hm_shape[1] - 0.5) * 2.

            smplx_joint_cam_29, _ = convert_kps(smplx_joint_cam[np.newaxis], src='smplx', dst='hybrik_29')
            smplx_joint_cam_29 = smplx_joint_cam_29[0] - smplx_joint_cam_29[0, 0]
            d_29 = smplx_joint_cam_29[:, 2, np.newaxis]

            score_joints = np.concatenate([x_29_norm, y_29_norm, d_29], axis=1).astype(np.float32)
            
            inputs = {'img': img}
            targets = {}
            meta_info = {'img_path': img_path, 
                        'ann_id': data['ann_id']}
            gen_output = {'score_joints': score_joints}
            return inputs, targets, meta_info, gen_output

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__  == 'SHAPY':
            inputs = {'img': img}
            if cfg.shapy_eval_split == 'val':
                targets = {'smplx_shape': smplx_shape}
            else:
                targets = {}
            meta_info = {'img_path': img_path}
            return inputs, targets, meta_info

        else:
            raise NotImplementedError