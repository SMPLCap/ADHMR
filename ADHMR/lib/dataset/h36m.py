"""Human3.6M dataset."""
import json
import os

import cv2
import joblib
import numpy as np
import torch.utils.data as data
import pickle as pk
import torch

from utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from utils.pose_utils import cam2pixel, pixel2cam, reconstruction_error
from utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam)
from tqdm import tqdm
from utils.transforms import flip_xyz_joints_3d


class H36MDataset(data.Dataset):
    """ Human3.6M smpl dataset. 17 Human3.6M joints + 29 SMPL joints

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/h36m'
        Path to the h36m dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 17 + 29
    num_thetas = 24
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_29 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    )
    joints_name_14 = (
        'R_Ankle', 'R_Knee', 'R_Hip',           # 2
        'L_Hip', 'L_Knee', 'L_Ankle',           # 5
        'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
        'Neck', 'Head'
    )

    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

    block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']
    #block_list = []

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/h36m',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):
        self._cfg = cfg
        self.protocol = cfg.dataset.protocol
        self.validate = not train                   # apply sampling.interval to self.db

        self._ann_file = os.path.join(
            root, 'annotations', ann_file + f'_protocol_{self.protocol}.json')
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self._det_bbox_file = getattr(cfg.dataset.set_list[0], 'DET_BOX', None)
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

        self.eval_14 = cfg.sampling.eval_14

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self._depth_dim = cfg.dataset.depth_dim


        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.lshoulder_idx_17 = self.joints_name_17.index('L_Shoulder')
        self.rshoulder_idx_17 = self.joints_name_17.index('R_Shoulder')
        self.root_idx_smpl = self.joints_name_29.index('pelvis')
        self.lshoulder_idx_29 = self.joints_name_29.index('left_shoulder')
        self.rshoulder_idx_29 = self.joints_name_29.index('right_shoulder')
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
                train=self._train, add_dpg=self._dpg,
                scale_mult=1)

    def __getitem__(self, idx):
        # get image id
        img_path = self.db['img_path'][idx]
        img_id = self.db['img_id'][idx]
        abs_id = self.db['abs_id'][idx]

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
        # for DPO dataset construction
        # return img, target, abs_id, bbox, img_path

    def __len__(self):
        return len(self.db['img_path'])

    def load_pt(self):
        if os.path.exists(self._ann_file + '.pt'):
            db = joblib.load(self._ann_file + '.pt', 'r')
        else:
            self._save_pt()
            torch.distributed.barrier()
            db = joblib.load(self._ann_file + '.pt', 'r')
        assert len(db['img_id']) == len(db['img_path'])
        abs_id_list = np.arange(len(db['img_id']))
        db.update({'abs_id': abs_id_list})
        for k in db.keys():
            db[k] = db[k][::max(self.interval,1)]
        # if self.validate:
        #     for k in db.keys():
        #         db[k] = db[k][::max(self.interval,1)]
        return db

    def _save_pt(self):
        _items, _labels = self._load_jsons()
        keys = list(_labels[0].keys())
        _db = {}
        for k in keys:
            _db[k] = []

        print(f'Generating Human3.6M pt: {len(_labels)}...')
        for obj in _labels:
            for k in keys:
                _db[k].append(np.array(obj[k]))

        _db['img_path'] = _items
        for k in keys:
            _db[k] = np.stack(_db[k])
            assert _db[k].shape[0] == len(_labels)
        joblib.dump(_db, self._ann_file + '.pt')

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)
        # iterate through the annotations
        bbox_scale_list = []
        det_bbox_set = {}
        if self._det_bbox_file is not None:
            bbox_list = json.load(open(os.path.join(
                self._root, 'annotations', self._det_bbox_file + f'_protocol_{self.protocol}.json'), 'r'))
            for item in bbox_list:
                image_id = item['image_id']
                det_bbox_set[image_id] = item['bbox']

        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v
            skip = False
            for name in self.block_list:
                if name in ann['file_name']:
                    skip = True
            if skip:
                continue
            abs_path = os.path.join(self._root, 'images', ann['file_name'])

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            if self._det_bbox_file is not None:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                    bbox_xywh_to_xyxy(det_bbox_set[ann['file_name']]), width, height)
            else:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                    bbox_xywh_to_xyxy(ann['bbox']), width, height)

            f, c = np.array(ann['cam_param']['f'], dtype=np.float32), np.array(
                ann['cam_param']['c'], dtype=np.float32)

            joint_cam_17 = np.array(ann['h36m_joints']).reshape(17, 3)
            joint_cam = np.array(ann['smpl_joints'])
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)
            beta = np.array(ann['betas'])
            theta = np.array(ann['thetas']).reshape(self.num_thetas, 3)

            joint_img_17 = cam2pixel(joint_cam_17, f, c)
            joint_img_17[:, 2] = joint_img_17[:, 2] - joint_cam_17[self.root_idx_17, 2]
            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_img_29 = cam2pixel(joint_cam_29, f, c)
            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_cam_29[self.root_idx_smpl, 2]
            if 'joint_2d' in ann.keys():
                joint_2d = np.array(ann['joint_2d']).reshape(29,2)
            else:
                joint_2d = joint_cam_29[:,:2]
            joint_vis_17 = np.ones((17, 3))
            joint_vis_29 = np.ones((29, 3))

            root_cam = np.array(ann['root_coord'])


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
            
            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'width': width,
                'height': height,
                'joint_img_17': joint_img_17,
                'joint_vis_17': joint_vis_17,
                'joint_cam_17': joint_cam_17,
                'joint_relative_17': joint_relative_17,
                'joint_img_29': joint_img_29,
                'joint_vis_29': joint_vis_29,
                'joint_cam_29': joint_cam_29,
                'twist_phi': phi,
                'twist_weight': phi_weight,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': f,
                'c': c,
                'joint_2d': joint_2d
            })
            bbox_scale_list.append(max(xmax - xmin, ymax - ymin))

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
        return ((0, 3), (1, 4), (2, 5), (10, 13), (11, 14), (12, 15))
    
    @property
    def missing_joint(self):
        # 0：左手，1：右手，2：左腿，3：右腿
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

    def evaluate_uvd_24(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 24))      # joint error
        error_x = np.zeros((sample_num, 24))    # joint error
        error_y = np.zeros((sample_num, 24))    # joint error
        error_z = np.zeros((sample_num, 24))    # joint error
        # error for each sequence
        error_action = [[] for _ in range(len(self.action_name))]
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            f = self.db['f'][n]
            c = self.db['c'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24].copy()
            

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd_jts'][:24].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[2] + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            if self.protocol == 1:
                # rigid alignment for PA MPJPE (protocol #1)
                pred_3d_kpt = reconstruction_error(pred_3d_kpt, gt_3d_kpt)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': int(image_id), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'

        eval_summary = f'UVD_24 Protocol {self.protocol} error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)

        print(eval_summary)

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
        # error for each sequence
        error_action = {}
        error_action_gt = {}
        error = np.zeros((sample_num, 24))  # joint error
        error_align = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        # error for each sequence
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24].copy()

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
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            act = self.action_name[action_idx]
            if act not in error_action.keys():
                error_action[act] = []
                error_action_gt[act] = []
            error_action[act].append(error[n].copy().mean())
            error_action_gt[act].append(error_align[n].copy().mean())

            # prediction save
            
            pred_save.append({'image_id': int(image_id), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist(),'joint_uvd':preds[image_id]['uvd_29'].copy().tolist()})  # joint_cam is root-relative coordinate

        for key in error_action.keys():
            error_action[key] = np.array(error_action[key])
            error_action_gt[key] =np.array(error_action_gt[key])
        error_all_joint = np.array(error)
        error_all_joint_gt = np.array(error_align)
        error_dict = {
            'error_all_joint': np.array(error_all_joint),
            'error_all_joint_gt': np.array(error_all_joint_gt),
            'error_all_joint_action': error_action,
            'error_all_joint_gt_action':error_action_gt
        }

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return error_dict

    def evaluate_xyz_17(self, preds, result_dir,flip=False):
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = {'image_id':[], 'joint_cam':[],'joint_uvd':[]}
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
        error_action = {}
        error_action_gt = {}
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

            # rigid alignment for PA MPJPE
            # pred_3d_kpt_align = rigid_align(pred_3d_kpt.copy(), gt_3d_kpt.copy())
            pred_3d_kpt_align = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())
            # pred_3d_kpt_align = pred_3d_kpt_align - pred_3d_kpt_align[self.root_idx_17]

            # select eval 14 joints
            if self.eval_14:
                pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
                gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)
                pred_3d_kpt_align = np.take(pred_3d_kpt_align, self.EVAL_JOINTS, axis=0)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            act = self.action_name[action_idx]
            if act not in error_action.keys():
                error_action[act] = []
                error_action_gt[act] = []
            error_action[act].append(error[n].copy().mean())
            error_action_gt[act].append(error_align[n].copy().mean())

            # prediction save
            pred_save['image_id'].append(int(image_id))
            pred_save['joint_cam'].append(np.array(pred_3d_kpt_save))
            pred_save['joint_uvd'].append(np.array(preds[image_id]['uvd_29']))
            for key in preds[image_id].keys():
                if key in pred_save:
                    pred_save[key].append(np.array(preds[image_id][key]))
        for key in error_action.keys():
            error_action[key] = np.array(error_action[key])
            error_action_gt[key] =np.array(error_action_gt[key])
        error_all_joint = np.array(error)
        error_all_joint_gt = np.array(error_align)
        error_dict = {
            'error_all_joint': np.array(error_all_joint),
            'error_all_joint_gt': np.array(error_all_joint_gt),
            'error_all_joint_action': error_action,
            'error_all_joint_gt_action':error_action_gt,
            'PVE':np.array(pred_save['pve']),
            'score': np.array(pred_save['score'])
        }
        for key in pred_save.keys():
            pred_save[key] = np.array(pred_save[key]).tolist()
        # prediction save
        return error_dict,pred_save