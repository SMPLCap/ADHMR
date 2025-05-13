import os.path as osp
import numpy as np
import scipy.sparse as ssp
import cv2
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt

from utils.bbox import _box_to_center_scale, _center_scale_to_box,get_bbox,process_bbox, bbox_clip_xyxy, bbox_xywh_to_xyxy
from utils.transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy,
                          rotmat_to_quat_numpy, flip_twist)
from utils.pose_utils import cam2pixel, pixel2cam, reconstruction_error,pixel2cam_test, get_intrinsic_metrix
from utils.presets import (SimpleTransform3DSMPL,SimpleTransform3DSMPLCam)
from utils.draw import *
def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5 # fov: 55 degree

class DemoDataset(Dataset):
    def __init__(self,cfg, img_path_list, detection_list,debug=None):
        self.detection_list = detection_list
        self.img_path_list = img_path_list
        self._input_size = cfg.hrnet.image_size
        self._output_size = cfg.hrnet.heatmap_size

        self.human36_joint_num = 17
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.vertex_num = 6890
        self.bbox_3d_shape = getattr(cfg.dataset, 'bbox_3d_shape', (2000, 2000, 2000))
        self.depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32)
        self.drawpath = cfg.inference.out_dir
        self.input_type = cfg.inference.input_type

    def __len__(self):
        return len(self.detection_list)

    def __getitem__(self, idx):
        """
        self.detection_list: [[frame_id, x_min, y_min, x_max, y_max, pixel_root_x, pixel_root_y, depth]]
        """
        det_info = self.detection_list[idx]
        img_idx = int(det_info[0])
        img_path = self.img_path_list[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        ori_img_height, ori_img_width = img.shape[:2]
        focal = estimate_focal_length(ori_img_height, ori_img_width)
        focal_l = np.array([focal, focal])
        center_pt = np.array([ori_img_width/2 , ori_img_height/2])
        intrinsic_param = get_intrinsic_metrix(focal_l, center_pt, inv=True).astype(np.float32)
        img_name = img_path.split('/')[-1]

        bbox = det_info[1:5]
        
        bbox = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
        xmin, ymin, w, h = bbox
        # use the estimated root depth
        root_img = np.array([det_info[5], det_info[6], det_info[7]/1700*focal])
        root_cam = pixel2cam_test(root_img[None,:].copy(), focal_l, center_pt)
        '''
        plt.figure()
        plt.imshow(img)
        drawbbox(det_info[1:5])
        plt.scatter(cam2pixel(root_cam, focal_l, center_pt)[0,0],cam2pixel(root_cam, focal_l, center_pt)[0,1])
        plt.savefig(os.path.join(self.drawpath,'bbox',str(idx)+img_name))
        plt.close()
        '''

        center = (float(bbox[0] + 0.5*bbox[2]), float(bbox[1] + 0.5*bbox[3]))
        aspect_ratio = float(self._input_size[1]) / self._input_size[0]  # w / h
        scale_mult = 1.25
        center, scale = _box_to_center_scale(xmin, ymin, w, h, aspect_ratio, scale_mult=scale_mult)

        trans = get_affine_transform(center, scale, 0, (self._input_size[1], self._input_size[0])).astype(np.float32)
        trans_inv = get_affine_transform(center, scale, 0, (self._input_size[1], self._input_size[0]), inv=True).astype(np.float32)
        img = cv2.warpAffine(img, trans, (self._input_size[1], self._input_size[0]), flags=cv2.INTER_LINEAR)
        img = im_to_torch(img).float()
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)
        target = {
                'idx':idx,
                'img_idx': img_idx,
                'img_path': img_path,
                'img_name': img_name,
                'trans_inv': torch.from_numpy(trans_inv).float(),
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(root_cam[0].astype(np.float32)).float(),
                'depth_factor': torch.from_numpy(np.array([2000]).astype(np.float32)).float(),
                'bbox': torch.Tensor(bbox),
                'trans': torch.from_numpy(trans).float(),
                'f': torch.from_numpy(np.array(focal_l).astype(np.float32)).float(),
                'c': torch.from_numpy(np.array(center_pt).astype(np.float32)).float(),
            }

        return img, target, img_idx