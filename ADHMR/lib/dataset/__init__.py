# ------------------------------------------------------------------------------
# pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.h36m import H36MDataset as h36m
from dataset.hp3d import HP3D
from dataset.mscoco import Mscoco
from dataset.pw3d import PW3D
from dataset.mpii import MPII
from dataset.mix_dataset import MixDataset
from dataset.up3d import UP3D
from dataset.surreal import SURREAL