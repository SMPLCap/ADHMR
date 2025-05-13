import os
import os.path as osp
import numpy as np
import torch
import cv2
from .humandata import HumanDataset


class InstaVariety(HumanDataset):
    def __init__(self, cfg, transform, data_split):
        super(InstaVariety, self).__init__(cfg, transform, data_split)

        self._cfg = cfg

        self.datalist = []

        pre_prc_file = 'insta_variety_neural_annot_train.npz'
        if self.data_split == 'train':
            filename = getattr(self._cfg, 'filename', pre_prc_file)
        else:
            # raise ValueError('InstaVariety test set is not support')
            filename = getattr(self._cfg, 'filename', pre_prc_file)

        self.img_dir = osp.join(self._cfg.data_dir, 'InstaVariety')
        self.annot_path = osp.join(self._cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(self._cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(self._cfg, 'use_cache', False)
        self.img_shape = (224, 224)  # (h, w)
        self.cam_param = {}

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
                train_sample_interval=getattr(self._cfg, f'{self.__class__.__name__}_train_sample_interval', 1),
                test_sample_interval=getattr(self._cfg, f'{self.__class__.__name__}_test_sample_interval', 10))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)