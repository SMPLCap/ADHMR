import os
import logging
import time
import glob
import datetime
import json
import pickle
import torch
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from utils.filter_hub import *
from utils.pose_utils import *
from utils.diff_utils import *
from utils.transforms import *
from utils.inference import *
from utils.draw import *
from utils.function import *
from dataset.demo_dataset import DemoDataset
from models.layers.smpl.SMPL import SMPL_layer

from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose.utils.utils import load_backbone_validate
import virtualpose.models as det_models
import virtualpose.dataset as det_dataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class Inference(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.num_joints = config.hyponet.num_joints
        self.num_twists = config.hyponet.num_twists
        self.num_item = self.num_joints*3+self.num_twists*2
        self.image_size = np.array(config.hrnet.image_size)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.multihypo_n = config.sampling.multihypo_n
        self.topk = config.training.scorenet.topk

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alpha = alphas_cumprod
        h36m_jregressor = np.load('data/smpl/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            'data/smpl/SMPL_NEUTRAL.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        ).to(self.device)
        

    def validate(self):
        args, config = self.args, self.config
        
        rank = torch.distributed.get_rank()
        model, model_cond, __, __, __, __, __, __, __, __, __ = get_model(config, is_train=False, resume = True, resume_path = config.training.scorenet.test_path)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)
        model_cond = nn.parallel.DistributedDataParallel(model_cond, device_ids=[args.device], output_device=args.local_rank)
        model_score, model_score_cond, __, __, __, __, epoch, step, __ = get_model_score(config, is_train=False, resume = True, resume_path = self.config.sampling.ckpt)
        model_score = nn.parallel.DistributedDataParallel(model_score, device_ids=[args.device], output_device=args.local_rank)
        model_score_cond = nn.parallel.DistributedDataParallel(model_score_cond, device_ids=[args.device], output_device=args.local_rank)
        state = dict(model = model, model_cond = model_cond, model_score = model_score, epoch = epoch, model_score_cond = model_score_cond)

        KST = datetime.timezone(datetime.timedelta(hours=8))
        if config.inference.input_type == 'video':
            dname = config.inference.input_name
        else:
            dname = 'image'
        config.inference.out_dir = os.path.join(config.inference.out_dir,str(datetime.datetime.now(tz=KST))[5:-16]+dname)

        list_name = [config.inference.input_type,'mesh']
        for name in list_name:
            path = os.path.join(config.inference.out_dir,name)
            if os.path.exists(path) and rank == 0:
                shutil.rmtree(path)
            os.makedirs(path,exist_ok=True)

        img_path_list, img_dir, fps = get_image_path(config)
        
        
        virtualpose_name = 'VirtualPose' 
        det_update_config(f'{virtualpose_name}/configs/images/images_inference.yaml')
        
        cur_path = config.inference.det_dir
        img_dir = img_dir

        det_model = eval('det_models.multi_person_posenet.get_multi_person_pose_net')(det_cfg, is_train=False)
        with torch.no_grad():
            det_model = torch.nn.DataParallel(det_model,device_ids=[rank])

        pretrained_file = osp.join(cur_path, f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED)
        state_dict = torch.load(pretrained_file)
        new_state_dict = {k:v for k, v in state_dict.items() if 'backbone.pose_branch.' not in k}
        det_model.module.load_state_dict(new_state_dict, strict = False)
        pretrained_file = osp.join(cur_path, f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED_BACKBONE)
        det_model = load_backbone_validate(det_model, pretrained_file)

        # prepare detection dataset
        infer_dataset = det_dataset.images(
            det_cfg, img_dir, focal_length=1700, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
            ]))
        infer_loader = torch.utils.data.DataLoader(
            infer_dataset,
            batch_size=config.inference.det_bs,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            drop_last=False,)
        
        det_model.eval()

        max_person = 0
        detection_all = []
        valid_frame_idx_all = []
        img_list_all = []
        with torch.no_grad():
            f_start = -1
            for _, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(infer_loader, dynamic_ncols=True)):
                for k in meta.keys():
                    try:
                        meta[k] = meta[k].to(self.device)
                    except Exception:
                        pass
                    inputs = inputs.to(self.device)
                    targets_2d =targets_2d.to(self.device)
                    targets_3d =targets_3d.to(self.device)
                    weights_2d = weights_2d.to(self.device)
                    input_AGR = input_AGR.to(self.device)
                _, _, output, _, _ = det_model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                                weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
                det_results, n_person, valid_frame_idx,img_list,f_start = output2original_scale(meta, output, start=f_start+1)
                detection_all += det_results
                valid_frame_idx_all += valid_frame_idx
                img_list_all += img_list
                max_person = max(n_person, max_person)
        

        # list to array
        detection_all = np.array(detection_all)

        valid_dataset = DemoDataset(config,img_list_all, detection_all)
        logger.info('valid_dataset.length:{}'.format(len(valid_dataset)))
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=config.sampling.batch_size,
                shuffle=False,
                num_workers=config.dataset.workers,
                sampler=valid_sampler,
                pin_memory=True,
                drop_last=False)

        state['dataset'] = valid_dataset
        state['dataloader'] =  valid_loader
        state['dataset_type'] = 'TEST'
        state['detection_all'] = detection_all
        state['fps'] = fps
        state['max_person'] = max_person

        out = self.sample(state)
    

    def sample(self, state):
        args, config = self.args, self.config
        
        model = state['model']
        model_cond = state['model_cond']
        model_score = state['model_score']
        model_score_cond = state['model_score_cond']
        dataset = state['dataset']
        dataloader = state['dataloader']
        epoch = state['epoch']
        dataset_type = state['dataset_type']

        rank = torch.distributed.get_rank()
        
        multi_n = self.multihypo_n
        with torch.no_grad():
            pred = {}
            for i in range(multi_n+2):
                pred[i] = {}
            for i, (inps, labels, img_ids) in enumerate(tqdm(dataloader, ncols=100)):
                n = inps.size(0)
                output = {}
                for k, _ in labels.items():
                    try:
                        labels[k] = labels[k].to(self.device)
                    except Exception:
                        pass
                input = inps.float().to(self.device)
                scale = torch.tensor([self.image_size[0],self.image_size[1],dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                labels['trans_inv']= labels['trans_inv'].unsqueeze(1).repeat(1,multi_n+2,1,1).view(-1,2,3)
                labels['intrinsic_param']= labels['intrinsic_param'].unsqueeze(1).repeat(1,multi_n+2,1,1).view(-1,3,3)
                labels['joint_root']= labels['joint_root'].unsqueeze(1).repeat(1,multi_n+2,1).view(-1,3)

                state['input'] = input.clone()
                state['scale'] = scale.clone()
                output, score_input = self.gen_mesh(state,multi_n)
                output['pred_shape'] = output['pred_shape'].unsqueeze(1).repeat(1,multi_n+2,1).view(-1,10)
                output['pred_joints'] = output['pred_joints'].view(-1,self.num_joints,3)
                output['pred_twist'] = output['pred_twist'].view(-1,self.num_twists,2)
                
                cond_feature,__ = model_score_cond(input.clone())
                
                score_input = torch.cat([score_input['joint'].view(n*multi_n,-1),score_input['twist'].view(n*multi_n,-1)],dim=1).to(self.device)
                score = model_score(score_input,cond_feature)

                
                score = score.view(n,multi_n)
                joints = output['pred_joints'].view(n,multi_n,self.num_joints,3)
                twist = output['pred_twist'].view(n,multi_n,self.num_twists,2)
                idx_score = score.contiguous().argsort(dim=1)[:,-self.topk:].contiguous().view(-1)
                idx_bs = torch.arange(n).repeat(self.topk).sort()[0]
                
                select_score = score[idx_bs,idx_score].view(n,self.topk)
                select_score = torch.nn.functional.softmax(select_score,dim=-1).unsqueeze(-1)


                select_joints = joints[idx_bs,idx_score].view(n,self.topk,self.num_joints,3).mean(dim=1)


                select_twist = twist[idx_bs,idx_score].view(n,self.topk,self.num_twists,2)
                select_twist = select_twist/(torch.norm(select_twist, dim=-1, keepdim=True) + 1e-8)
                select_angle = torch.arctan(select_twist[:,:,:,1]/select_twist[:,:,:,0]) 
                flag = (torch.cos(select_angle)*select_twist[:,:,:,0])<0
                select_angle = select_angle + flag*np.pi
                assert ((torch.cos(select_angle)-select_twist[:,:,:,0]).abs()>1e-6).sum() ==0
                assert ((torch.sin(select_angle)-select_twist[:,:,:,1]).abs()>1e-6).sum() ==0
                select_angle = select_angle.mean(dim=1)
                select_twist = select_twist.mean(dim=1)
                select_twist = torch.cat([torch.cos(select_angle).unsqueeze(-1),torch.sin(select_angle).unsqueeze(-1)],dim=-1)

                output['pred_joints'] = output['pred_joints'].view(n,multi_n,self.num_joints,3)
                mean_joint = output['pred_joints'].mean(dim=1).unsqueeze(1)
                output['pred_joints'] = torch.cat([select_joints.unsqueeze(1),output['pred_joints'],mean_joint],dim=1).view(-1,self.num_joints,3)
                output['pred_twist'] = output['pred_twist'].view(n,multi_n,self.num_twists,2)
                mean_twist = output['pred_twist'].mean(dim=1).unsqueeze(1)
                output['pred_twist'] = torch.cat([select_twist.unsqueeze(1),output['pred_twist'],mean_twist],dim=1).view(-1,self.num_twists,2)

                score = torch.cat([torch.zeros(n,1).to(score.device)+score.max()+50,score.view(n,multi_n),torch.zeros(n,1).to(score.device)],dim=-1)
                score = score.view(n,multi_n+2).cpu().data.numpy()
            
                pred_shape = output['pred_shape'].clone().cpu().data.numpy()
                pred_uvd_jts_29 = trans_back(output['pred_joints'].clone(),labels['trans_inv'].clone()).cpu().data.numpy()
                pred_twist = output['pred_twist'].clone().cpu().data.numpy()
                
                pred_shape = pred_shape.reshape(n,multi_n+2,10)
                pred_twist = pred_twist.reshape(n,multi_n+2,23,2)
                pred_uvd_jts_29 = pred_uvd_jts_29.reshape(n,multi_n+2,29,3)

                output_final = get_estimates(config, labels, output, self.smpl)

                pred_joints = output_final['pred_joints'].clone().cpu().data.numpy() * 2
                pred_joints = pred_joints.reshape(n,multi_n+2,29,3)
                pred_uvd_jts_24 = output_final['uvd_24'].cpu().data.numpy().reshape(n,multi_n+2,-1,2)

                # vertice
                pred_mesh = output_final['pred_vertices'].reshape(n,multi_n+2,-1,3)
                draw_mesh = pred_mesh.view(n,multi_n+2,-1,3).cpu().data.numpy()
                pred_mesh = pred_mesh.cpu().data.numpy()
                pred_xyz_jts_17 = output_final['pred_xyz_jts_17'].reshape(n,multi_n+2, 17, 3).cpu().data.numpy()
                
                f = labels['f'].cpu().data.numpy()
                c = labels['c'].cpu().data.numpy()
                root_img = pred_uvd_jts_29[:,0,0,:2]
                root_cam = labels['joint_root'].cpu().data.numpy() 
                idx = labels['idx'].cpu().data.numpy() 
                results = {'pred_mesh':pred_mesh[:,:self.config.inference.draw_num],
                          'pose_root':root_cam,
                          'focal_l':f,
                          'center_pt':c,
                          'pred_root_xy_img':root_img,
                          'img_name':labels['img_name'],
                          'idx':idx,
                          'img_path':labels['img_path'],
                          'fps':state['fps'],
                          'max_person':state['max_person']}
                if not self.config.inference.input_type == 'video' and self.config.inference.render:
                    visualize(results,config.inference.draw_num,save_path= os.path.join(self.config.inference.out_dir,'image'),detection_all=state['detection_all'],input_type=self.config.inference.input_type)
                    
                
                labels['img_idx']= labels['img_idx'].cpu().data.numpy()
                for k in range(multi_n+2):
                    for j in range(pred_xyz_jts_17.shape[0]):
                        item_i = {'xyz_17': pred_xyz_jts_17[j][k],
                                'score':score[j][k],
                                'uvd_29':pred_uvd_jts_29[j][k],
                                'uvd_24':pred_uvd_jts_24[j][k],
                                'focal_l':f[j],
                                'center_pt':c[j],
                                'idx':idx[j],
                                'img_idx':labels['img_idx'][j],
                                'img_path':labels['img_path'][j],
                                'pred_mesh': pred_mesh[j][k]
                                }
                        pred[k][int(idx[j])] = item_i
                

        torch.distributed.barrier()
        save_path=os.path.join(self.config.inference.out_dir,'result')
        os.makedirs(save_path,exist_ok=True)
        path = save_path+'_'+str(rank)+'.pkl'
        with open(path, 'wb') as fid:
            pickle.dump(pred, fid, pickle.HIGHEST_PROTOCOL)
        print('dump the file',path)
        torch.distributed.barrier()
        pred = {}
        if rank==0:
            print('gpu num: {}'.format(args.world_size))
            for i in range(multi_n+2):
                pred[i] = {}
            for r in range(self.args.world_size):
                path = save_path+'_'+str(r)+'.pkl'
                with open(path, 'rb') as fid:
                    pred_i = pickle.load(fid)
                    print('load the file',path)
                for midx in range(multi_n+2):
                    pred[midx].update(pred_i[midx])
                os.remove(path)
            path = os.path.join(save_path,'output.pkl')
            with open(path, 'wb') as fid:
                 pickle.dump(pred, fid, pickle.HIGHEST_PROTOCOL)
            
            results = {'pred_mesh':[],
                       'focal_l':[],
                       'center_pt':[],
                       'img_path':[],
                       'uvd_29':[],
                       'uvd_24':[],
                       'img_idx':[]
            }
            key_list = list(pred[0].keys())
            key_list.sort()
            for k in key_list:
                for k_r in results.keys():
                    results[k_r].append(pred[0][k][k_r])
            for k_r in results.keys():
                results[k_r] = np.array(results[k_r])
            results['fps']=state['fps']
            results['max_person']=state['max_person']
            results['img_path_video'] = dataset.img_path_list

            if self.config.inference.render and self.config.inference.input_type == 'video':
                results['video_name'] = config.inference.input_name[:-4]
                visualize(results,config.inference.draw_num,save_path= os.path.join(self.config.inference.out_dir,'video'),detection_all=state['detection_all'],input_type=self.config.inference.input_type)
        torch.distributed.barrier()
        return pred
        

    def sample_pose(self, xj,xt, model,content=None, last=True,gen_multi=True):
        
        args, config = self.args, self.config
        if self.config.diffusion.skip_type == "uniform":
            skip = self.num_timesteps // self.config.diffusion.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.config.diffusion.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), self.config.diffusion.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        if self.num_timesteps-1 not in list(seq):
            seq = list(seq)+[self.num_timesteps-1]

        xs = generalized_steps(xj,xt, seq, model, self.betas.to(self.device), eta=self.config.diffusion.eta,ctx =content, gen_multi=gen_multi)
        x = xs
        if last:
            x = [x[0][0][-1],x[0][1][-1]]
        return x

    def gen_mesh(self, state, multi_n):
        model = state['model']
        model_cond = state['model_cond']
        scale = state['scale']
        input = state['input']

        gen_output = {}
        score_input = {}

        ctx , pred_shape, __ = model_cond(input)
        gen_output['pred_shape'] = pred_shape

        n = input.shape[0]
        if self.config.sampling.zero and self.config.sampling.multihypo_n==1:
            xj = torch.zeros(n*multi_n, self.num_joints*3).to(self.device)
            xt = torch.zeros(n*multi_n, self.num_twists*2).to(self.device)
        else:
            xj = torch.randn(n,multi_n, self.num_joints*3).to(self.device)
            xt = torch.randn(n,multi_n, self.num_twists*2).to(self.device)
            xj[:,0] = 0
            xj = xj.view(n*multi_n, self.num_joints*3).to(self.device)
            xt[:,0]=0
            xt = xt.view(n*multi_n, self.num_twists*2).to(self.device)
        
        pred_j,pred_t= self.sample_pose(xj=xj,xt=xt,model=model,content=ctx,gen_multi=True)
        gen_output['pred_joints'] = denormalize_pose_cuda(pose = pred_j, which=self.config.hyponet.norm,scale=scale,mean_and_std=self.config.diffusion.scale).to(self.device).view(n,multi_n,self.num_joints,3)
        gen_output['pred_twist'] = pred_t.view(n, multi_n,self.num_twists,2) / self.config.diffusion.scale

        score_input['joint'] = normalize_pose_cuda(pose = gen_output['pred_joints'].clone().view(-1,self.num_joints,3), which='scale_t',scale=scale*self.config.diffusion.scale).to(self.device).view(n,multi_n,self.num_joints*3)
        score_input['twist'] = gen_output['pred_twist'].clone()

        return gen_output, score_input