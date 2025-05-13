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
    get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset
from humandata_scorer_test import HumanDataset_Scorer_Test


class GTA_Human2(HumanDataset):
    def __init__(self, transform, data_split):
        super(GTA_Human2, self).__init__(transform, data_split)

        filename = 'gta_human2multiple_230406_04000_0.npz'
        self.img_dir = osp.join(cfg.data_dir, 'GTA_Human2')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {
            'focal': (1158.0337, 1158.0337),  # (fx, fy)
            'princpt': (960, 540)  # (cx, cy)
        }
        self.test_sample_interval = getattr(cfg, f'{self.__class__.__name__}_test_sample_interval', 100)

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 10),
                test_sample_interval=self.test_sample_interval)
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)

class GTA_Human2_SCORER_TEST(HumanDataset_Scorer_Test):
    def __init__(self, transform, data_split):
        super(GTA_Human2_SCORER_TEST, self).__init__(transform, data_split)
        
        filename = 'gta_human2multiple_230406_04000_0.npz'
        self.img_dir = osp.join(cfg.data_dir, 'GTA_Human2')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache_scorer_eval', 
                                         f'pointwise_GTA_Human2_test_interval_100_multin_1.npz')
        self.use_cache = True
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {
            'focal': (1158.0337, 1158.0337),  # (fx, fy)
            'princpt': (960, 540)  # (cx, cy)
        }
        assert self.data_split == 'test', "Only support Scorer test!"

        self.use_cache = True

        self.test_sample_interval = getattr(cfg, f'{self.__class__.__name__}_test_sample_interval', 100)

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            raise NotImplementedError