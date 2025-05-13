"""MS COCO Human keypoint dataset."""
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

class Mscoco(data.Dataset):
    """ COCO Person dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/coco'
        Path to the ms coco dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
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
                 root='./data/coco',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'annotations', ann_file)
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
        img_id = self.db['img_id'][idx] #int(os.path.splitext(os.path.basename(img_path))[0])

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
        _items, _labels = self._load_jsons_smpl()
        keys = list(_labels[0].keys())
        _db = {}
        for k in keys:
            _db[k] = []

        print(f'Generating COCO pt: {len(_labels)}...')
        for obj in _labels:
            for k in keys:
                _db[k].append(np.array(obj[k]))

        _db['img_path'] = _items
        for k in keys:
            _db[k] = np.stack(_db[k])
            assert _db[k].shape[0] == len(_labels)

        joblib.dump(_db, self._ann_file + '.pt')
    


    def _load_jsons_smpl(self):
        items = []
        labels = []
        cnt = 0
        with open(self._ann_file,'rb') as fid:
            db_temp = json.load(fid)
        for i in range(len(db_temp)):
            abs_path = os.path.join(self._root, 'images', db_temp[i]['img_path'])
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
            joint_vis_29 = np.zeros((29, 3))

            joint_cam_vis = np.array(ann['joint_cam_valid'])
            if joint_cam_vis.size == 24 * 3:
                joint_vis_29[:24, :] = joint_cam_vis.reshape(24, 3)
            else:
                joint_vis_29 = joint_cam_vis.reshape(29, 3)
            
            
            root_cam = joint_cam_29[0]
            
            beta = np.array(ann['smpl_param']['shape']).reshape(10)
            theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)
            
            
            if 'angle_twist' in ann.keys(): #2
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
                    'joint_2d':joint_2d,
                }
            )
            cnt +=1
            
        return items, labels
    
    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        _coco = COCO(self._ann_file)
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self._root, dirname, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d,
                # 'segmentation': obj['segmentation'],
                'keypoints': obj['keypoints'],
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

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

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 3), (1, 4), (2, 5), (10, 13), (11, 14), (12, 15))
