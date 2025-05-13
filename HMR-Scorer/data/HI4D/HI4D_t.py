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
from humandata import HumanDataset, TemporalHumanDataset


class HI4D_t(TemporalHumanDataset):
    def __init__(self, transform, data_split):
        super(HI4D_t, self).__init__(transform, data_split)

        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', 'hi4d_train_240205_098.npz')
        else:
            raise ValueError('test set is not support')

        self.seqlen = getattr(cfg, 'seqlen', 16)
        self.overlap = getattr(cfg, 'overlap', 0.)
        self.stride = int(self.seqlen * (1-self.overlap))
        
        self.img_dir = osp.join(cfg.data_dir, 'hi4d')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', filename)
        self.annot_chunk_cache = osp.join(cfg.data_dir, 'cache', 'chunks', 'hi4d_train_240205_098.pkl')
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = None # (h, w)
        self.cam_param = {}
        self.train_sample_interval = getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1)
        self.test_sample_interval = getattr(cfg, f'{self.__class__.__name__}_test_sample_interval', 1)

        # # check image shape
        # img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        # img_shape = cv2.imread(img_path).shape[:2]
        # assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
            self.chunks = self.load_chunk(self.annot_chunk_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist, self.chunks = self.get_chunk()
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
                self.save_chunk(self.annot_chunk_cache, self.chunks)