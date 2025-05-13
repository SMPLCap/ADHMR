"""3DPW dataset."""
import json
import os

import cv2
import joblib
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
import torch

from utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from utils.pose_utils import pixel2cam, reconstruction_error
from utils.pose_utils import cam2pixel, pixel2cam, reconstruction_error
from utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam)
from tqdm import tqdm
from utils.transforms import flip_xyz_joints_3d


class PW3D(data.Dataset):
    """ 3DPW dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/pw3d'
        Path to the PW3D dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 17
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
    joints_name_14 = (
        'R_Ankle', 'R_Knee', 'R_Hip',           # 2
        'L_Hip', 'L_Knee', 'L_Ankle',           # 5
        'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
        'Neck', 'Head'
    )
    skeleton = (
        (1, 0), (2, 1), (3, 2),  # 2
        (4, 0), (5, 4), (6, 5),  # 5
        (7, 0), (8, 7),  # 7
        (9, 8), (10, 9),  # 9
        (11, 7), (12, 11), (13, 12),  # 12
        (14, 7), (15, 14), (16, 15),  # 15
    )

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/pw3d',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):
        self._cfg = cfg
        self.validate = not train

        self._ann_file = os.path.join(root, 'annotations',ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
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

        self._sigma = cfg.dataset.sigma

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self.num_joints = cfg.hyponet.num_joints

        self.num_joints_half_body = cfg.dataset.num_joints_half_body
        self.prob_half_body = cfg.dataset.prob_half_body

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self._depth_dim = cfg.dataset.depth_dim
        self.eval_14  = cfg.sampling.eval_14


        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.lshoulder_idx_17 = self.joints_name_17.index('L_Shoulder')
        self.rshoulder_idx_17 = self.joints_name_17.index('R_Shoulder')
        self.root_idx_smpl = self.joints_name_24.index('pelvis')
        self.lshoulder_idx_24 = self.joints_name_24.index('left_shoulder')
        self.rshoulder_idx_24 = self.joints_name_24.index('right_shoulder')
        self.interval  = cfg.sampling.interval

        self.db = self.load_pt()

        self.transformation = SimpleTransform3DSMPL(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=self._occlusion,
            flip = self._flip,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg)

    def __getitem__(self, idx):
        # get image id
        img_path = self.db['img_path'][idx]
        img_id = self.db['img_id'][idx]

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.db.keys():
            label[k] = self.db[k][idx].copy()
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
        if self.validate:
            for k in db.keys():
                db[k] = db[k][::max(self.interval,1)]
        return db

    def _save_pt(self):
        if self.validate:
            _items, _labels = self._lazy_load_json()
        else:
            _items, _labels = self._lazy_load_json_2()
        keys = list(_labels[0].keys())
        _db = {}
        for k in keys:
            _db[k] = []

        print(f'Generating 3DPW pt: {len(_labels)}...')
        for obj in _labels:
            for k in keys:
                _db[k].append(np.array(obj[k]))

        _db['img_path'] = _items
        for k in keys:
            _db[k] = np.stack(_db[k])
            assert _db[k].shape[0] == len(_labels)

        joblib.dump(_db, self._ann_file + '.pt')
    
    def _lazy_load_json_2(self):
        """Load all image paths and labels from json annotation files into buffer."""

        items = []
        labels = []

        db = COCO(self._ann_file)
        cnt = 0

        for aid in db.anns.keys():
            ann = db.anns[aid]

            img_id = ann['image_id']

            img = db.loadImgs(img_id)[0]
            width, height = img['width'], img['height']

            sequence_name = img['sequence']
            img_name = img['file_name']
            abs_path = os.path.join(
                self._root, 'imageFiles', sequence_name, img_name)

            beta = np.array(ann['smpl_param']['shape']).reshape(10)
            theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)

            x, y, w, h = ann['bbox']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)
            if xmin > xmax - 5 or ymin > ymax - 5:
                continue

            f = np.array(img['cam_param']['focal'], dtype=np.float32)
            c = np.array(img['cam_param']['princpt'], dtype=np.float32)

            joint_cam_17 = np.array(ann['h36m_joints'], dtype=np.float32).reshape(17, 3)*1000
            joint_vis_17 = np.ones((17, 3))
            joint_img_17 = np.zeros((17, 3))

            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_cam = np.array(ann['smpl_joint_cam']) #1
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)
            
            joint_img_29_0 = cam2pixel(joint_cam_29, f, c)
            joint_img_29_0[:, 2] = joint_img_29_0[:, 2] - joint_cam_29[self.root_idx_smpl, 2]


            joint_img = np.array(ann['smpl_joint_img'], dtype=np.float32)
            if joint_img.size == 24 * 3:
                joint_img = joint_img.reshape(24,3)
                joint_img[:,2] = joint_img[:,2]*1000
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :] = joint_img.reshape(24, 3)
            elif joint_img.size == 24 * 2:
                joint_img = joint_img.reshape(24,2)
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :2] = joint_img
            else:
                joint_img_29 = joint_img.reshape(29, 3)

            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_img_29[self.root_idx_smpl, 2]

            joint_vis_24 = np.ones((24, 3))
            joint_vis_29 = np.ones((29, 3))

            root_cam = joint_cam_29[self.root_idx_smpl]

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
            
            if 'joint_2d' in ann.keys():
                joint_2d = np.array(ann['joint_2d']).reshape(29,2)
            else:
                joint_2d = joint_img_29_0[:,:2]

            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': cnt,
                'img_path': abs_path,
                'img_name': img_name,
                'width': width,
                'height': height,
                'joint_img_17': joint_img_17,
                'joint_vis_17': joint_vis_17,
                'joint_cam_17': joint_cam_17,
                'joint_relative_17': joint_relative_17,
                'joint_img_29': joint_img_29_0, #4
                'joint_vis_29': joint_vis_29,
                'joint_cam_29': joint_cam_29,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'twist_phi': phi, #3
                'twist_weight': phi_weight,
                'f': f,
                'c': c,
                'test':joint_img_29_0,
                'joint_2d': joint_2d
            })
            cnt += 1

        return items, labels

    def _lazy_load_json(self):
        """Load all image paths and labels from json annotation files into buffer."""

        items = []
        labels = []

        db = COCO(self._ann_file)
        cnt = 0

        for aid in db.anns.keys():
            ann = db.anns[aid]

            img_id = ann['image_id']

            img = db.loadImgs(img_id)[0]
            width, height = img['width'], img['height']

            sequence_name = img['sequence']
            img_name = img['file_name']
            abs_path = os.path.join(
                self._root, 'imageFiles', sequence_name, img_name)

            beta = np.array(ann['smpl_param']['shape']).reshape(10)
            theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)

            x, y, w, h = ann['bbox']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)
            if xmin > xmax - 5 or ymin > ymax - 5:
                continue

            f = np.array(img['cam_param']['focal'], dtype=np.float32)
            c = np.array(img['cam_param']['princpt'], dtype=np.float32)

            joint_cam_17 = np.array(ann['h36m_joints'], dtype=np.float32).reshape(17, 3)*1000
            joint_vis_17 = np.ones((17, 3))
            joint_img_17 = np.zeros((17, 3))

            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_cam = np.array(ann['smpl_joint_cam'])*1000
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)
            
            joint_img_29_0 = cam2pixel(joint_cam_29, f, c)
            joint_img_29_0[:, 2] = joint_img_29_0[:, 2] - joint_cam_29[self.root_idx_smpl, 2]

            joint_img = np.array(ann['smpl_joint_img'], dtype=np.float32)
            if joint_img.size == 24 * 3:
                joint_img = joint_img.reshape(24,3)
                joint_img[:,2] = joint_img[:,2]*1000
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :] = joint_img.reshape(24, 3)
            elif joint_img.size == 24 * 2:
                joint_img = joint_img.reshape(24,2)
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :2] = joint_img
            else:
                joint_img_29 = joint_img.reshape(29, 3)

            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_img_29[self.root_idx_smpl, 2]

            joint_vis_24 = np.ones((24, 3))
            joint_vis_29 = np.zeros((29, 3))
            joint_vis_29[:24, :] = joint_vis_24

            root_cam = joint_cam_29[self.root_idx_smpl]
            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': cnt,
                'img_path': abs_path,
                'img_name': img_name,
                'width': width,
                'height': height,
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
                'f': f,
                'c': c,
                'test':joint_img_29_0,
                'tran_sl': np.array(ann['smpl_param']['trans']),
            })
            cnt += 1

        return items, labels

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

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 1), (2, 3), (4, 5), (7, 8), (9, 10), (11, 12))

    @property
    def missing_joint(self):
        return ((17,19,21,23),(16,18,20,22),(2,5,8,11),(1,4,7,10))

    def evaluate_uvd_24(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            f = self.db['f'][n]
            c = self.db['c'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24, :].copy()

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd_jts'][:24, :].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[2] + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]

            # prediction save
            pred_save.append({'img_name': str(img_name), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_kp = np.mean(error, axis=0) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000
        metric = 'MPJPE'

        eval_summary = f'UVD_24 error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)
        print(f'UVD_24 error per joint: {tot_err_kp}')

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_24(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_align = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24, :].copy()

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_24'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # rigid alignment for PA MPJPE
            pred_3d_kpt_align = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]

            # prediction save
            pred_save.append({'img_name': str(img_name), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist(),'joint_uvd':preds[image_id]['uvd_29'].copy().tolist()})  # joint_cam is root-relative coordinate

        error_all_joint = np.array(error)
        error_all_joint_gt = np.array(error_align)
        error_dict = {
            'error_all_joint': np.array(error_all_joint),
            'error_all_joint_gt': np.array(error_all_joint_gt),
        }

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return error_dict

    def evaluate_xyz_17(self, preds, result_dir,flip=False):
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = {'img_name':[], 'joint_cam':[],'joint_uvd':[]}
        for key in next(iter(preds.items()))[1].keys():
            pred_save[key] = []
        if self.eval_14:
            error = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
            error_align = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
            error_x = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
            error_y = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
            error_z = np.zeros((sample_num, len(self.EVAL_JOINTS)))
        else:
            error = np.zeros((sample_num, 17))  # joint error
            error_align = np.zeros((sample_num, 17))  # joint error
            error_x = np.zeros((sample_num, 17))  # joint error
            error_y = np.zeros((sample_num, 17))  # joint error
            error_z = np.zeros((sample_num, 17))  # joint error

        
        # error for each sequence
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_relative_17'][n].copy()
            imgwidth =self.db['width'][n].copy()
            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_17'].copy() * self.bbox_3d_shape[2]
            if flip:
                pred_3d_kpt = flip_xyz_joints_3d(pred_3d_kpt,self.joint_pairs_17)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]
            pred_3d_kpt_save = pred_3d_kpt.copy()

            # select eval 14 joints
            if self.eval_14:
                pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
                gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]

            # prediction save
            pred_save['img_name'].append(str(img_name))
            pred_save['joint_cam'].append(np.array(pred_3d_kpt_save))
            pred_save['joint_uvd'].append(np.array(preds[image_id]['uvd_29']))
            for key in preds[image_id].keys():
                if key in pred_save:
                    pred_save[key].append(np.array(preds[image_id][key]))
        error_all_joint = np.array(error)
        error_all_joint_gt = np.array(error_align)
        error_dict = {
            'error_all_joint': np.array(error_all_joint),
            'error_all_joint_gt': np.array(error_all_joint_gt),
            'PVE':np.array(pred_save['pve']),
            'score': np.array(pred_save['score'])
        }

        for key in pred_save.keys():
            pred_save[key] = np.array(pred_save[key]).tolist()
        # prediction save
        pred_save.pop('img_name')
        return error_dict,pred_save
