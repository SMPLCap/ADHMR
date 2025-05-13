import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import subprocess
import glob
from collections import defaultdict
import imageio
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored
from torchvision import transforms
from torch.utils.data import DataLoader
from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose.utils.utils import load_backbone_validate
import virtualpose.models as det_models
import virtualpose.dataset as det_dataset
from models.layers.smpl.SMPL import SMPL_layer
from utils.draw import *
from utils.pose_utils import *
import shutil
h36m_jregressor = np.load('data/smpl/J_regressor_h36m.npy')
smpl = SMPL_layer(
    'data/smpl/SMPL_NEUTRAL.pkl',
    h36m_jregressor=h36m_jregressor,
    dtype=torch.float32
)

def output2original_scale(meta, output, vis=False,start=0):
    img_paths, trans_batch = meta['image'], meta['trans']
    bbox_batch, depth_batch, roots_2d = output['bboxes'], output['depths'], output['roots_2d']

    scale = torch.tensor((det_cfg.NETWORK.IMAGE_SIZE[0] / det_cfg.NETWORK.HEATMAP_SIZE[0], \
                        det_cfg.NETWORK.IMAGE_SIZE[1] / det_cfg.NETWORK.HEATMAP_SIZE[1]), \
                        device=bbox_batch.device, dtype=torch.float32)
    
    det_results = []
    valid_frame_idx = []
    img_list = []
    max_person = 0
    for i, img_path in enumerate(img_paths):
        if vis:
            img = cv2.imread(img_path)
        frame_id = i+start
        trans = trans_batch[i].to(bbox_batch[i].device).float()
        
        n_person = 0
        for bbox, depth, root_2d in zip(bbox_batch[i], depth_batch[i], roots_2d[i]):
            if torch.all(bbox == 0):
                break
            bbox = (bbox.view(-1, 2) * scale[None, [1, 0]]).view(-1)
            root_2d *= scale[[1, 0]]
            bbox_origin = inverse_affine_transform_pts_cuda(bbox.view(-1, 2), trans).reshape(-1)
            roots_2d_origin = inverse_affine_transform_pts_cuda(root_2d.view(-1, 2), trans).reshape(-1)

            # frame_id, x_min, y_min, x_max, y_max, pixel_root_x, pixel_root_y, depth
            det_results.append([frame_id] + bbox_origin.cpu().numpy().tolist() + roots_2d_origin.cpu().numpy().tolist() + depth.cpu().numpy().tolist())
            img_list.append(img_path)

            if vis:
                img = cv2.putText(img, '%.2fmm'%depth, (int(bbox_origin[0]), int(bbox_origin[1] - 5)),\
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                img = cv2.rectangle(img, (int(bbox_origin[0]), int(bbox_origin[1])), (int(bbox_origin[2]), int(bbox_origin[3])), \
                    (255, 0, 0), 1)
                img = cv2.circle(img, (int(roots_2d_origin[0]), int(roots_2d_origin[1])), 5, (0, 0, 255), -1)
            n_person += 1

        if vis:
           cv2.imwrite(f'origin_det_{i}.jpg', img)
        max_person = max(n_person, max_person)
        if n_person:
            valid_frame_idx.append(frame_id)

        
    return det_results, max_person, valid_frame_idx,img_list,frame_id

def visualize(results,draw_num,save_path,detection_all=None,input_type='image'):
    pred_mesh = results['pred_mesh']#*1000  # (N*T, V, 3)
    focal_l = results['focal_l']
    center_pt = results['center_pt']
    if input_type=='video':
        img_idx_list = results['img_idx']
        img_path_list = results['img_path_video']
        max_person = results['max_person']
        video_name = results['video_name']
        videowriter = imageio.get_writer(osp.join(save_path, f"{video_name}_results_in_2d.mp4"), fps=results['fps'])
        frame_num = max(img_idx_list)
        for i in tqdm(range(frame_num)):
            chosen_mask = img_idx_list == i
            chosen_idx = np.where(chosen_mask>0)[0]
            if chosen_idx.shape[0] == 0:
                continue
            pred_mesh_T = pred_mesh[chosen_idx]
            focal_T = focal_l[chosen_idx[0]]
            center_pt_T = center_pt[chosen_idx[0]]
            img_path = img_path_list[chosen_idx[0]]
            img = cv2.imread(img_path)
            ori_img_height, ori_img_width = img.shape[:2]
            rgb, depth = render_mesh(ori_img_height, ori_img_width, pred_mesh_T/1000.0, smpl.faces, {'focal': focal_T, 'princpt': center_pt_T})
            valid_mask = (depth > 0)[:,:,None] 
            rendered_img = rgb * valid_mask + img[:,:,::-1] * (1-valid_mask)
            cv2.imwrite(osp.join(save_path, f"{video_name}_results_in_2d.jpg"), rendered_img.astype(np.uint8)[...,::-1])
            videowriter.append_data(rendered_img.astype(np.uint8))
        videowriter.close()
    else:
        img_path_list = results['img_path']
        img_name = results['img_name']
        idx_list =results['idx']
        for mesh_idx in tqdm(range(pred_mesh.shape[0])):
            img_path = img_path_list[mesh_idx]
            img = cv2.imread(img_path)
            ori_img_height, ori_img_width = img.shape[:2]
            dirpath = osp.join(save_path,img_name[mesh_idx],str(idx_list[mesh_idx]))
            if os.path.exists(dirpath) and os.path.isdir(dirpath):
                shutil.rmtree(dirpath)
            for i in range(draw_num):
                pred_mesh_T = pred_mesh[mesh_idx,i]
                rgb, depth = render_mesh(ori_img_height, ori_img_width, [pred_mesh_T/1000], smpl.faces, {'focal': focal_l[mesh_idx], 'princpt': center_pt[mesh_idx]})

                valid_mask = (depth > 0)[:,:,None] 
                rendered_img = rgb * valid_mask + img[:,:,::-1] * (1-valid_mask)
                save_path_img = osp.join(save_path,img_name[mesh_idx],str(idx_list[mesh_idx]),f"idx{idx_list[mesh_idx]} hypo{i}.jpg")
                os.makedirs(osp.join(save_path,img_name[mesh_idx],str(idx_list[mesh_idx])),exist_ok=True)
                cv2.imwrite(save_path_img, rendered_img.astype(np.uint8)[...,::-1])

class DeltaDepth(nn.Module):
    def __init__(self,bs):
        super(DeltaDepth, self).__init__()
        self.delta_d = nn.Parameter(torch.zeros(bs,1),requires_grad=True)
    def forward(self,depth):
        return depth+self.delta_d
    
class DeltaDepthBeta(nn.Module):
    def __init__(self,bs):
        super(DeltaDepthBeta, self).__init__()
        self.delta_d = nn.Parameter(torch.zeros(bs,1),requires_grad=True)
        self.delta_b = nn.Parameter(torch.zeros(bs,1,10),requires_grad=True)
    def forward(self,depth,beta):
        return depth+self.delta_d,self.delta_b+beta

def video_to_images(vid_file, img_folder=None):
    cap = cv2.VideoCapture(vid_file)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    command = ['ffmpeg',
               '-i', vid_file,
               '-r', str(fps),
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    print(f'Images saved to \"{img_folder}\"')
    return fps

def get_image_path(args):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff','.JPG']
    if args.inference.input_type == "image":
        img_dir = args.inference.img_dir
        img_path_list = [osp.join(img_dir, f) for f in os.listdir(img_dir) 
                     if osp.isfile(osp.join(img_dir, f)) and osp.splitext(f)[1].lower() in image_extensions]
        fps = -1
    elif args.inference.input_type == "video":
        video_path = osp.join(args.inference.img_dir,args.inference.input_name)
        basename = osp.basename(video_path).split('.')[0]
        img_dir = osp.join(osp.abspath(args.inference.img_dir), basename)
        os.makedirs(img_dir, exist_ok=True)
        fps = video_to_images(video_path, img_folder=img_dir)

        # get all image paths
        img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
        img_path_list.extend(glob.glob(osp.join(img_dir, '*.png')))
        img_path_list.sort()
    else:
        assert 0, 'only support image/video input type'
    return img_path_list, img_dir, fps

def get_estimates(config, labels, output, smpl):
    with torch.enable_grad():
        init_depth = labels['joint_root'][:,2].clone().view(-1,config.sampling.multihypo_n+2).detach()
        n = init_depth.shape[0]
        depth_model = DeltaDepth(n).cuda(init_depth.device)
        depth_model.train()
        optim_list = [{"params":depth_model.delta_d,"lr":config.inference.optim_lr}]
        optimizer = torch.optim.Adam(optim_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.inference.step_size, gamma=config.inference.gamma)
        for i in range(config.inference.optim_step):
            optimizer.zero_grad()
            new_label = {}
            for k in labels.keys():
                try:
                    new_label[k] = labels[k].detach().clone()
                except Exception:
                    pass
            new_output = {}
            for k in output.keys():
                new_output[k] = output[k].detach().clone()
            new_label['joint_root'][:,2] = depth_model(init_depth).view(-1)
            output_final = process_output(new_output,new_label,smpl,process=True)
            output_final['loss'].backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
    
    return output_final