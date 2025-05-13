import os
import os.path as osp
import numpy as np
import cv2
from utils.transforms_humandata import world2cam, cam2pixel, rigid_align
from .humandata import HumanDataset


class HI4D(HumanDataset):
    def __init__(self, cfg, transform, data_split):
        super(HI4D, self).__init__(cfg, transform, data_split)

        self._cfg = cfg

        if self.data_split == 'train':
            filename = getattr(self._cfg, 'filename', 'hi4d_train_240205_098.npz')
        else:
            filename = getattr(self._cfg, 'filename', 'hi4d_train_240205_098.npz')
            # raise ValueError('test set is not support')

        self.seqlen = getattr(self._cfg, 'seqlen', 16)
        self.overlap = getattr(self._cfg, 'overlap', 0.)
        self.stride = int(self.seqlen * (1-self.overlap))
        
        self.img_dir = osp.join(self._cfg.data_dir, 'hi4d')
        self.annot_path = osp.join(self._cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(self._cfg.data_dir, 'cache', filename)
        self.annot_chunk_cache = osp.join(self._cfg.data_dir, 'cache', 'chunks', 'hi4d_train_240205_098.pkl')
        self.use_cache = getattr(self._cfg, 'use_cache', False)
        # self.img_shape = None # (h, w)
        #[DEBUG]
        self.img_shape = (1280, 940)
        
        self.cam_param = {}
        # self.train_sample_interval = getattr(self._cfg, f'{self.__class__.__name__}_train_sample_interval', 10)
        # self.test_sample_interval = getattr(self._cfg, f'{self.__class__.__name__}_test_sample_interval', 1)

        # # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        print(img_path)
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
            # self.chunks = self.load_chunk(self.annot_chunk_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            
            # self.datalist, self.chunks = self.get_chunk()

            self.datalist = self.load_data(
                train_sample_interval=getattr(self._cfg, f'{self.__class__.__name__}_train_sample_interval', 10),
                test_sample_interval=getattr(self._cfg, f'{self.__class__.__name__}_test_sample_interval', 10))
            
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
                # self.save_chunk(self.annot_chunk_cache, self.chunks)