import math
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..bbox import _box_to_center_scale, _center_scale_to_box, get_bbox, get_bbox_filter, process_bbox
from ..transforms import (addDPG, affine_transform, inverse_affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy,
                          rotmat_to_quat_numpy, flip_twist)
from ..pose_utils import get_intrinsic_metrix
import logging
logging.basicConfig(level=logging.INFO)
s_coco_2_smpl_jt = [
    -1, 11, 12,
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

s_coco_2_h36m_jt = [
    -1,
    -1, 13, 15,
    -1, 14, 16,
    -1, -1,
    0, -1,
    5, 7, 9,
    6, 8, 10
]

s_coco_2_smpl_jt_2d = [
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

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]


class SimpleTransform3DSMPL_Humandata(object):
    """Generation of cropped input person, pose coords, smpl parameters.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, color_factor, occlusion,flip, add_dpg,
                 input_size, output_size,depth_dim, bbox_3d_shape,
                 rot, sigma, train, scale_mult=1.25, two_d=False):
        if two_d:
            self._joint_pairs = dataset.joint_pairs
        else:
            self._joint_pairs_17 = dataset.joint_pairs_17
            self._joint_pairs_24 = dataset.joint_pairs_24
            self._joint_pairs_29 = dataset.joint_pairs_29
        self.missing_joint = dataset.missing_joint

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._flip = flip
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        self.bbox_3d_shape = dataset.bbox_3d_shape
        self._scale_mult = scale_mult
        self.two_d = two_d

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids
       
    def test_transform(self, src, bbox):
        if isinstance(src, str):
            src = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)

        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        return img, bbox
    
    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (num_joints, self._heatmap_size[1], self._heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self._sigma * 3 

        for joint_id in range(num_joints):
            feat_stride = np.array(self._input_size) / np.array(self._heatmap_size)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)   
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self._heatmap_size[0] or ul[1] >= self._heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self._sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
    
    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[0]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_uvd_target_generator(self, joints_3d, num_joints, patch_height, patch_width):

        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_xyz_target_generator(self, joints_3d, joints_3d_vis, num_joints):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d_vis[:, 0]
        target_weight[:, 1] = joints_3d_vis[:, 1]
        target_weight[:, 2] = joints_3d_vis[:, 2]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0] / self.bbox_3d_shape[0]
        target[:, 1] = joints_3d[:, 1] / self.bbox_3d_shape[1]
        target[:, 2] = joints_3d[:, 2] / self.bbox_3d_shape[2]

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        if self.two_d:
            raise NotImplementedError
        else:
            bbox = list(label['bbox'])                                      # (x1, y1, x2, y2)
            joint_img_17 = label['joint_img_17'].copy()                     # all zeros
            joint_relative_17 = label['joint_relative_17'].copy()           # relative joint cam: [17, 3]
            joints_vis_17 = label['joint_vis_17'].copy()
            joint_img_29 = label['joint_img_29'].copy()                     # joints img 29: [29, 3]
            joint_cam_29 = label['joint_cam_29'].copy()
            joints_vis_29 = label['joint_vis_29'].copy()                    # joints img 29 conf: [29, 3]
            if 'joint_2d' in label.keys():
                joint_2d = label['joint_2d'].copy()
            else:
                joint_2d = np.zeros((29,2),dtype = np.float32)
            fx, fy = label['f'].copy()

            beta = label['beta'].copy()
            theta = label['theta'].copy()
            if 'twist_phi' in label.keys():
                twist_phi = label['twist_phi'].copy()
                twist_weight = label['twist_weight'].copy()
            else:
                twist_phi = np.zeros((23, 2))
                twist_weight = np.zeros((23, 2))

            gt_joints_17 = np.zeros((17, 3, 2), dtype=np.float32)
            gt_joints_17[:, :, 0] = joint_img_17.copy()
            gt_joints_17[:, :, 1] = joints_vis_17.copy()
            gt_joints_29 = np.zeros((29, 3, 2), dtype=np.float32)
            gt_joints_29[:, :, 0] = joint_img_29.copy()
            gt_joints_29[:, :, 1] = joints_vis_29.copy()
            gt_joints_2d = np.zeros((29, 2, 2), dtype=np.float32)
            gt_joints_2d[:,:,0] = joint_2d.copy()
            gt_joints_2d[:,:,1] = 1

            imgwidth, imght = label['width'], label['height']
            assert imgwidth == src.shape[1] and imght == src.shape[0]

            input_size = self._input_size
            
            # bbox scale and crop
            if self._add_dpg and self._train:
                bbox = addDPG(bbox, imgwidth, imght)

            # [DEBUG] abandon using joints to calculate bbox; use the original bbox
            # bbox = get_bbox_filter(joint_img_29[:24])
            bbox = np.array(bbox)
            bbox[2:] = bbox[2:] - bbox[:2]
            bbox_ = list(bbox)

            # # Vis joint_img_29
            # for i in range (joint_img_29.shape[0]):
            #     # import pdb; pdb.set_trace()
            #     kps = joint_img_29[i]
            #     # import pdb;pdb.set_trace()
            #     image = cv2.circle(src.copy(), (int(kps[0]),int(kps[1])), radius=2, color=(0, 255, 0), thickness=2)
            # cv2.imwrite('joint_img_29.png', image)

            bbox = process_bbox(bbox_, aspect_ratio=self._aspect_ratio, scale=1.0)
            xmin, ymin, w, h = bbox

            # Vis bbox
            # cv2.imwrite('bbox.png', cv2.rectangle(src, (int(bbox[0]),int(bbox[1])), (int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1])), (0, 0, 255), thickness=4))

            center, scale = _box_to_center_scale(
                xmin, ymin, w, h, self._aspect_ratio, scale_mult=self._scale_mult)
            # center, scale = _box_to_center_scale(
            #     xmin, ymin, w, h, self._aspect_ratio, scale_mult=self._scale_mult)
            xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

            # half body transform
            if self._train and (np.sum(joints_vis_17[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    gt_joints_17[:, :, 0], joints_vis_17
                )

                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            # rescale
            if self._train:
                sf = self._scale_factor
                scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            else:
                scale = scale * 1.0

            # rotation
            if self._train:
                rf = self._rot
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            else:
                r = 0

            # occlusion
            if self._train and self._occlusion and random.random() <= 0.8:
                while True:
                    area_min = 0.0
                    area_max = 0.3
                    synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                    ratio_min = 0.5
                    ratio_max = 1 / 0.5
                    synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                    synth_h = math.sqrt(synth_area * synth_ratio)
                    synth_w = math.sqrt(synth_area / synth_ratio)
                    synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                    synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                    if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                        synth_xmin = int(synth_xmin)
                        synth_ymin = int(synth_ymin)
                        synth_w = int(synth_w)
                        synth_h = int(synth_h)
                        src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                        break
            
            joints_17_uvd = gt_joints_17
            joints_29_uvd = gt_joints_29
            joints_2d_uvd = gt_joints_2d
            joints_17_xyz = joint_relative_17
            joints_24_xyz = joint_cam_29 - joint_cam_29[0, :].copy()
            joints_24_xyz = joints_24_xyz.copy()

            # flip
            if random.random() > 0.5 and self._train and self._flip:
                # src, fliped = random_flip_image(src, px=0.5, py=0)
                # if fliped[0]:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]

                joints_17_uvd = flip_joints_3d(joints_17_uvd, imgwidth, self._joint_pairs_17)
                joints_29_uvd = flip_joints_3d(joints_29_uvd, imgwidth, self._joint_pairs_29)
                joints_2d_uvd = flip_joints_3d(joints_2d_uvd, imgwidth, self._joint_pairs_29)
                joints_17_xyz = flip_xyz_joints_3d(joints_17_xyz, self._joint_pairs_17)
                joints_24_xyz = flip_xyz_joints_3d(joints_24_xyz, self._joint_pairs_24)
                theta = flip_thetas(theta, self._joint_pairs_24)
                twist_phi, twist_weight = flip_twist(twist_phi, twist_weight, self._joint_pairs_24)
                center[0] = imgwidth - center[0] - 1
            

            theta_rot_mat = batch_rodrigues_numpy(theta)
            theta_quat = rotmat_to_quat_numpy(theta_rot_mat).reshape(24 * 4)

            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
            intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
            joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
            depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
            

            # affine transform
            for i in range(17):
                if joints_17_uvd[i, 0, 1] > 0.0:
                    joints_17_uvd[i, 0:2, 0] = affine_transform(joints_17_uvd[i, 0:2, 0], trans)


            for i in range(29):
                if joints_29_uvd[i, 0, 1] > 0.0:
                    joints_29_uvd[i, 0:2, 0] = affine_transform(joints_29_uvd[i, 0:2, 0], trans)
                joints_2d_uvd[i,:,0] = affine_transform(joints_2d_uvd[i, :, 0], trans)

            target_smpl_weight = torch.ones(1).float()
            theta_24_weights = np.ones((24, 4))

            theta_24_weights = theta_24_weights.reshape(24 * 4)

            # generate training targets
            """
            target_uvd_29, target_weight_29 = self.generate_heatmap(joints_29_uvd[:,:,0], joints_29_uvd[:,:,1])
            target_xyz_17, target_weight_17 = self._integral_xyz_target_generator(joints_17_xyz, joints_vis_17, 17)
            target_xyz_24, target_weight_24 = self._integral_xyz_target_generator(joints_24_xyz, joints_vis_29[:24, :], 24)
            target_weight_29 *= joints_vis_29.reshape(-1)
            target_weight_24 *= joints_vis_29[:24, :].reshape(-1)
            target_weight_17 *= joints_vis_17.reshape(-1)
            """
            bbox = _center_scale_to_box(center, scale)

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        if self.two_d:
            raise NotImplementedError
        else:
            output = {
                'image': img,
                'target_theta': torch.from_numpy(theta_quat).float(),                   # smpl theta in quanterion: [96,]
                'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
                'target_beta': torch.from_numpy(beta).float(),                          # smpl shape params: [10,]
                'target_smpl_weight': target_smpl_weight,
                'joints_uvd_29':torch.from_numpy(joints_29_uvd[:,:,0]).float(),         # joints img 29: [29, 3]
                'joints_2d': torch.from_numpy(joints_2d_uvd[:,:,0]).float(),
                'joints_vis_29':torch.from_numpy(joints_29_uvd[:,:,1]).float(),         # joints img 29 confidence: [29, 3]
                'joints_xyz_24':torch.from_numpy(joints_24_xyz[:24,:]).float(),
                'joints_vis_24':torch.from_numpy(joints_vis_29[:24, :]).float(),
                'joints_xyz_17':torch.from_numpy(joints_17_xyz).float(),                # relative joint cam: [17, 3]
                'joints_vis_17':torch.from_numpy(joints_vis_17).float(),                # joint cam conf: [17, 3]
                'scale': torch.from_numpy(scale).float(),
                'trans_inv': torch.from_numpy(trans_inv).float(),                       # [2, 3]
                'trans': torch.from_numpy(trans).float(),                               # [2, 3]
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(joint_root).float(),                     # root joint cam: [3,]; same as the first row of label['joint_cam_29']
                'depth_factor': torch.from_numpy(depth_factor).float(),
                'bbox': torch.Tensor(bbox),
                'target_twist': torch.from_numpy(twist_phi).float(),
                'target_twist_weight': torch.from_numpy(twist_weight).float(),
            }
        return output

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale
import math
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..bbox import _box_to_center_scale, _center_scale_to_box, get_bbox, get_bbox_filter, process_bbox
from ..transforms import (addDPG, affine_transform, inverse_affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy,
                          rotmat_to_quat_numpy, flip_twist)
from ..pose_utils import get_intrinsic_metrix
import logging
logging.basicConfig(level=logging.INFO)
s_coco_2_smpl_jt = [
    -1, 11, 12,
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

s_coco_2_h36m_jt = [
    -1,
    -1, 13, 15,
    -1, 14, 16,
    -1, -1,
    0, -1,
    5, 7, 9,
    6, 8, 10
]

s_coco_2_smpl_jt_2d = [
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

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]


class SimpleTransform3DSMPL_Humandata(object):
    """Generation of cropped input person, pose coords, smpl parameters.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, color_factor, occlusion,flip, add_dpg,
                 input_size, output_size,depth_dim, bbox_3d_shape,
                 rot, sigma, train, scale_mult=1.25, two_d=False):
        if two_d:
            self._joint_pairs = dataset.joint_pairs
        else:
            self._joint_pairs_17 = dataset.joint_pairs_17
            self._joint_pairs_24 = dataset.joint_pairs_24
            self._joint_pairs_29 = dataset.joint_pairs_29
        self.missing_joint = dataset.missing_joint

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._flip = flip
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        self.bbox_3d_shape = dataset.bbox_3d_shape
        self._scale_mult = scale_mult
        self.two_d = two_d

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids
       
    def test_transform(self, src, bbox):
        if isinstance(src, str):
            src = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)

        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        return img, bbox
    
    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (num_joints, self._heatmap_size[1], self._heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self._sigma * 3 

        for joint_id in range(num_joints):
            feat_stride = np.array(self._input_size) / np.array(self._heatmap_size)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)   
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self._heatmap_size[0] or ul[1] >= self._heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self._sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
    
    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[0]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_uvd_target_generator(self, joints_3d, num_joints, patch_height, patch_width):

        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_xyz_target_generator(self, joints_3d, joints_3d_vis, num_joints):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d_vis[:, 0]
        target_weight[:, 1] = joints_3d_vis[:, 1]
        target_weight[:, 2] = joints_3d_vis[:, 2]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0] / self.bbox_3d_shape[0]
        target[:, 1] = joints_3d[:, 1] / self.bbox_3d_shape[1]
        target[:, 2] = joints_3d[:, 2] / self.bbox_3d_shape[2]

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        if self.two_d:
            raise NotImplementedError
        else:
            bbox = list(label['bbox'])                                      # (x1, y1, x2, y2)
            joint_img_17 = label['joint_img_17'].copy()                     # all zeros
            joint_relative_17 = label['joint_relative_17'].copy()           # relative joint cam: [17, 3]
            joints_vis_17 = label['joint_vis_17'].copy()
            joint_img_29 = label['joint_img_29'].copy()                     # joints img 29: [29, 3]
            joint_cam_29 = label['joint_cam_29'].copy()
            joints_vis_29 = label['joint_vis_29'].copy()                    # joints img 29 conf: [29, 3]
            if 'joint_2d' in label.keys():
                joint_2d = label['joint_2d'].copy()
            else:
                joint_2d = np.zeros((29,2),dtype = np.float32)
            fx, fy = label['f'].copy()

            beta = label['beta'].copy()
            theta = label['theta'].copy()
            if 'twist_phi' in label.keys():
                twist_phi = label['twist_phi'].copy()
                twist_weight = label['twist_weight'].copy()
            else:
                twist_phi = np.zeros((23, 2))
                twist_weight = np.zeros((23, 2))

            gt_joints_17 = np.zeros((17, 3, 2), dtype=np.float32)
            gt_joints_17[:, :, 0] = joint_img_17.copy()
            gt_joints_17[:, :, 1] = joints_vis_17.copy()
            gt_joints_29 = np.zeros((29, 3, 2), dtype=np.float32)
            gt_joints_29[:, :, 0] = joint_img_29.copy()
            gt_joints_29[:, :, 1] = joints_vis_29.copy()
            gt_joints_2d = np.zeros((29, 2, 2), dtype=np.float32)
            gt_joints_2d[:,:,0] = joint_2d.copy()
            gt_joints_2d[:,:,1] = 1

            imgwidth, imght = label['width'], label['height']
            assert imgwidth == src.shape[1] and imght == src.shape[0]

            input_size = self._input_size
            
            # bbox scale and crop
            if self._add_dpg and self._train:
                bbox = addDPG(bbox, imgwidth, imght)

            # [DEBUG] abandon using joints to calculate bbox; use the original bbox
            # bbox = get_bbox_filter(joint_img_29[:24])
            bbox = np.array(bbox)
            bbox[2:] = bbox[2:] - bbox[:2]
            bbox_ = list(bbox)

            # # Vis joint_img_29
            # for i in range (joint_img_29.shape[0]):
            #     # import pdb; pdb.set_trace()
            #     kps = joint_img_29[i]
            #     # import pdb;pdb.set_trace()
            #     image = cv2.circle(src.copy(), (int(kps[0]),int(kps[1])), radius=2, color=(0, 255, 0), thickness=2)
            # cv2.imwrite('joint_img_29.png', image)

            bbox = process_bbox(bbox_, aspect_ratio=self._aspect_ratio, scale=1.0)
            xmin, ymin, w, h = bbox

            # Vis bbox
            # cv2.imwrite('bbox.png', cv2.rectangle(src, (int(bbox[0]),int(bbox[1])), (int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1])), (0, 0, 255), thickness=4))

            center, scale = _box_to_center_scale(
                xmin, ymin, w, h, self._aspect_ratio, scale_mult=self._scale_mult)
            # center, scale = _box_to_center_scale(
            #     xmin, ymin, w, h, self._aspect_ratio, scale_mult=self._scale_mult)
            xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

            # half body transform
            if self._train and (np.sum(joints_vis_17[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    gt_joints_17[:, :, 0], joints_vis_17
                )

                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            # rescale
            if self._train:
                sf = self._scale_factor
                scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            else:
                scale = scale * 1.0

            # rotation
            if self._train:
                rf = self._rot
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            else:
                r = 0

            # occlusion
            if self._train and self._occlusion and random.random() <= 0.8:
                while True:
                    area_min = 0.0
                    area_max = 0.3
                    synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                    ratio_min = 0.5
                    ratio_max = 1 / 0.5
                    synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                    synth_h = math.sqrt(synth_area * synth_ratio)
                    synth_w = math.sqrt(synth_area / synth_ratio)
                    synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                    synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                    if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                        synth_xmin = int(synth_xmin)
                        synth_ymin = int(synth_ymin)
                        synth_w = int(synth_w)
                        synth_h = int(synth_h)
                        src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                        break
            
            joints_17_uvd = gt_joints_17
            joints_29_uvd = gt_joints_29
            joints_2d_uvd = gt_joints_2d
            joints_17_xyz = joint_relative_17
            joints_24_xyz = joint_cam_29 - joint_cam_29[0, :].copy()
            joints_24_xyz = joints_24_xyz.copy()

            # flip
            if random.random() > 0.5 and self._train and self._flip:
                # src, fliped = random_flip_image(src, px=0.5, py=0)
                # if fliped[0]:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]

                joints_17_uvd = flip_joints_3d(joints_17_uvd, imgwidth, self._joint_pairs_17)
                joints_29_uvd = flip_joints_3d(joints_29_uvd, imgwidth, self._joint_pairs_29)
                joints_2d_uvd = flip_joints_3d(joints_2d_uvd, imgwidth, self._joint_pairs_29)
                joints_17_xyz = flip_xyz_joints_3d(joints_17_xyz, self._joint_pairs_17)
                joints_24_xyz = flip_xyz_joints_3d(joints_24_xyz, self._joint_pairs_24)
                theta = flip_thetas(theta, self._joint_pairs_24)
                twist_phi, twist_weight = flip_twist(twist_phi, twist_weight, self._joint_pairs_24)
                center[0] = imgwidth - center[0] - 1
            

            theta_rot_mat = batch_rodrigues_numpy(theta)
            theta_quat = rotmat_to_quat_numpy(theta_rot_mat).reshape(24 * 4)

            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
            intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
            joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
            depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
            

            # affine transform
            for i in range(17):
                if joints_17_uvd[i, 0, 1] > 0.0:
                    joints_17_uvd[i, 0:2, 0] = affine_transform(joints_17_uvd[i, 0:2, 0], trans)


            for i in range(29):
                if joints_29_uvd[i, 0, 1] > 0.0:
                    joints_29_uvd[i, 0:2, 0] = affine_transform(joints_29_uvd[i, 0:2, 0], trans)
                joints_2d_uvd[i,:,0] = affine_transform(joints_2d_uvd[i, :, 0], trans)

            target_smpl_weight = torch.ones(1).float()
            theta_24_weights = np.ones((24, 4))

            theta_24_weights = theta_24_weights.reshape(24 * 4)

            # generate training targets
            """
            target_uvd_29, target_weight_29 = self.generate_heatmap(joints_29_uvd[:,:,0], joints_29_uvd[:,:,1])
            target_xyz_17, target_weight_17 = self._integral_xyz_target_generator(joints_17_xyz, joints_vis_17, 17)
            target_xyz_24, target_weight_24 = self._integral_xyz_target_generator(joints_24_xyz, joints_vis_29[:24, :], 24)
            target_weight_29 *= joints_vis_29.reshape(-1)
            target_weight_24 *= joints_vis_29[:24, :].reshape(-1)
            target_weight_17 *= joints_vis_17.reshape(-1)
            """
            bbox = _center_scale_to_box(center, scale)

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        if self.two_d:
            raise NotImplementedError
        else:
            output = {
                'image': img,
                'target_theta': torch.from_numpy(theta_quat).float(),                   # smpl theta in quanterion: [96,]
                'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
                'target_beta': torch.from_numpy(beta).float(),                          # smpl shape params: [10,]
                'target_smpl_weight': target_smpl_weight,
                'joints_uvd_29':torch.from_numpy(joints_29_uvd[:,:,0]).float(),         # joints img 29: [29, 3]
                'joints_2d': torch.from_numpy(joints_2d_uvd[:,:,0]).float(),
                'joints_vis_29':torch.from_numpy(joints_29_uvd[:,:,1]).float(),         # joints img 29 confidence: [29, 3]
                'joints_xyz_24':torch.from_numpy(joints_24_xyz[:24,:]).float(),
                'joints_vis_24':torch.from_numpy(joints_vis_29[:24, :]).float(),
                'joints_xyz_17':torch.from_numpy(joints_17_xyz).float(),                # relative joint cam: [17, 3]
                'joints_vis_17':torch.from_numpy(joints_vis_17).float(),                # joint cam conf: [17, 3]
                'scale': torch.from_numpy(scale).float(),
                'trans_inv': torch.from_numpy(trans_inv).float(),                       # [2, 3]
                'trans': torch.from_numpy(trans).float(),                               # [2, 3]
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(joint_root).float(),                     # root joint cam: [3,]; same as the first row of label['joint_cam_29']
                'depth_factor': torch.from_numpy(depth_factor).float(),
                'bbox': torch.Tensor(bbox),
                'target_twist': torch.from_numpy(twist_phi).float(),
                'target_twist_weight': torch.from_numpy(twist_weight).float(),
            }
        return output

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale
