"""UP3D Human keypoint dataset."""
import os

import cv2
import joblib
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
import json
import torch

from utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from utils.presets import SimpleTransform, SimpleTransformCam
from utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam)
from utils.pose_utils import cam2pixel, pixel2cam, reconstruction_error

class UP3D(data.Dataset):
    """ UP3D Person dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/up3d'
        Path to the ms up3d dataset.
    train: bool, default is True
        If true, will set as training mode.
    """
    CLASSES = ['person']
    num_joints = 17
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                   'left_shoulder', 'right_shoulder',                           # 6
                   'left_elbow', 'right_elbow',                                 # 8
                   'left_wrist', 'right_wrist',                                 # 10
                   'left_hip', 'right_hip',                                     # 12
                   'left_knee', 'right_knee',                                   # 14
                   'left_ankle', 'right_ankle')                                 # 16

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/up3d',
                 train=True,
                 dpg=False,
                 lazy_import=False):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._train = train
        self._dpg = dpg

        self.bbox_3d_shape = getattr(cfg.dataset, 'bbox_3d_shape', (2000, 2000, 2000))

        self._scale_factor = cfg.dataset.scale_factor
        self._color_factor = cfg.dataset.color_factor
        self._rot = cfg.dataset.rot_factor
        self._input_size = cfg.hrnet.image_size
        self._output_size = cfg.hrnet.heatmap_size

        self._occlusion = cfg.dataset.occlusion
        self._flip = cfg.dataset.flip

        self.eval_14 = cfg.sampling.eval_14
        self.root_idx_17 =0 

        self._sigma = cfg.dataset.sigma

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = cfg.dataset.num_joints_half_body
        self.prob_half_body = cfg.dataset.prob_half_body

        self._depth_dim = cfg.dataset.depth_dim



        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.transformation = SimpleTransform3DSMPL(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=False,
            flip = self._flip,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg)
       
        self.db = self.load_pt()

    def __getitem__(self, idx):
        # get image id
        img_path = self.db['img_path'][idx]
        img_id = self.db['img_id'][idx]

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.db.keys():
            try:
                label[k] = self.db[k][idx].copy()
            except AttributeError:
                label[k] = self.db[k][idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox, img_path

    def __len__(self):
        return len(self.db['img_path'])

    def load_pt(self):
        if os.path.exists(self._ann_file + '.pt'):
            db = joblib.load(self._ann_file + '.pt', 'r')
        else:
            self._save_pt()
            torch.distributed.barrier()
            db = joblib.load(self._ann_file + '.pt', 'r')

        return db

    def _save_pt(self):
        _items, _labels = self._load_jsons()
        keys = list(_labels[0].keys())
        _db = {}
        for k in keys:
            _db[k] = []

        print(f'Generating UP3D pt: {len(_labels)}...')
        for obj in _labels:
            for k in keys:
                _db[k].append(np.array(obj[k]))

        _db['img_path'] = _items
        for k in keys:
            _db[k] = np.stack(_db[k])
            assert _db[k].shape[0] == len(_labels)

        joblib.dump(_db, self._ann_file + '.pt')
    
    def _load_jsons(self):
        items = []
        labels = []
        cnt = 0
        with open(self._ann_file,'rb') as fid:
            db_temp = json.load(fid)
        for i in range(len(db_temp)):
            abs_path = os.path.join(self._root,db_temp[i]['img_path'])
            items.append(abs_path)
            ann = db_temp[i].copy()
            
            f = np.array(ann['f'], dtype=np.float32)
            c = np.array(ann['c'], dtype=np.float32)
            
            joint_cam_17 = np.array(ann['joint_cam_17'], dtype=np.float32).reshape(17, 3)
            joint_vis_17 = np.ones((17, 3))
            joint_img_17 = np.zeros((17, 3))
            joint_relative_17 = joint_cam_17 - joint_cam_17[0, :].reshape(1,3)
            
            joint_cam = np.array(ann['joint_cam_29']) #1
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)
            joint_img_29_0 = cam2pixel(joint_cam_29, f, c)
            joint_img_29_0[:, 2] = joint_img_29_0[:, 2] - joint_cam_29[0, 2]

            joint_img = np.array(ann['joint_img_29'], dtype=np.float32)
            joint_img[:,2] = joint_img[:,2]
            if joint_img.size == 24 * 3:
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :] = joint_img.reshape(24, 3)
            else:
                joint_img_29 = joint_img.reshape(29, 3)

            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_img_29[0, 2]
            
            joint_2d = joint_img_29[:,:2]

            joint_vis_24 = np.ones((24, 3))
            joint_vis_29 = np.ones((29, 3))
            
            assert np.sum((joint_img_29-joint_img_29_0)**2) < 1
            
            root_cam = joint_cam_29[0]
            
            beta = np.array(ann['smpl_param']['shape']).reshape(10)
            theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)
            
            
            if 'angle_twist' in ann.keys(): 
                twist = ann['angle_twist']
                angle = np.array(twist['angle'])
                cos = np.array(twist['cos'])
                sin = np.array(twist['sin'])
                assert (np.cos(angle) - cos < 1e-6).all(), np.cos(angle) - cos
                assert (np.sin(angle) - sin < 1e-6).all(), np.sin(angle) - sin
                phi = np.stack((cos, sin), axis=1)
                phi_weight = (angle > -10) * 1.0  
                phi_weight = np.stack([phi_weight, phi_weight], axis=1)
            else:
                phi = np.zeros((23, 2))
                phi_weight = np.zeros_like(phi)
            labels.append(
                {
                    'img_name':db_temp[i]['img_name'],
                    'img_id': cnt,
                    'bbox': np.array(db_temp[i]['bbox'],dtype=np.float32),
                    'width':db_temp[i]['width'],
                    'height':db_temp[i]['height'],
                    'joint_img_17': joint_img_17,
                    'joint_vis_17': joint_vis_17,
                    'joint_cam_17': joint_cam_17,
                    'joint_relative_17': joint_relative_17,
                    'joint_img_29': joint_img_29,
                    'joint_vis_29': joint_vis_29,
                    'joint_cam_29': joint_cam_29,
                    'beta': beta,
                    'theta': theta,
                    'root_cam': root_cam,
                    'twist_phi': phi, 
                    'twist_weight': phi_weight,
                    'f': f,
                    'c': c,
                    'joint_2d':joint_2d
                }
            )
            cnt+=1
        return items, labels
    
    
    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
    
    @property
    def missing_joint(self):
        return ((17,19,21,23),(16,18,20,22),(2,5,8,11),(1,4,7,10))
    
    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
    
    @property
    def joint_pairs_17(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    @property
    def joint_pairs_24(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    @property
    def joint_pairs_29(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
