import os
import sys
import logging
import time
import glob
import json
import cv2
import pickle
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record

from utils.filter_hub import *
from utils.pose_utils import *
from utils.diff_utils import *
from utils.function import get_optimizer, get_model, get_model_score, get_dataloader, process_pred
from utils.relation import *
from utils.cache import Cache
from models.layers.smpl.SMPL import SMPL_layer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def write_obj(path,point,face):
    with open(path,'w') as fid:
        for p in point:
            fid.write('v {} {} {}\n'.format(str(p[0]),str(p[1]),str(p[2])))
        for f in face:
            fid.write('f {} {} {}\n'.format(str(f[0]+1),str(f[1]+1),str(f[2]+1)))

class ScorenetTrainer(object):
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
        

    def train(self):
        args, config = self.args, self.config
        rank = torch.distributed.get_rank()
        
        num_models = len(config.training.scorenet.gen_path)
        model, model_cond = [], []
        for idx in range(num_models):
            model_i, model_cond_i, __, __, __, __, __, __, __, __, __ = get_model(config, is_train=False, resume = True, resume_path = config.training.scorenet.gen_path[idx])
            model.append(model_i)
            model_cond.append(model_cond_i)
        
        model_score, model_score_cond, ema_score, ema_score_cond, optimizer_score, optimizer_score_cond, start_epoch, step, loss_score = get_model_score(config, is_train=True, resume = config.training.resume_training, resume_path = self.config.training.resume_ckpt)
            
        model_score = nn.parallel.DistributedDataParallel(model_score, device_ids=[args.device], output_device=args.local_rank,find_unused_parameters=True)
        model_score_cond = nn.parallel.DistributedDataParallel(model_score_cond, device_ids=[args.device], output_device=args.local_rank,find_unused_parameters=True)
            
        for idx in range(num_models):
            for param in model_cond[idx].parameters():
                param.requires_grad = False
            for param in model[idx].parameters():
                param.requires_grad = False
        
    
        train_loaders, train_datasets, train_samplers = get_dataloader(config, is_train = True)
        train_loader = train_loaders['mix']
        train_dataset = train_datasets['mix']
        train_sampler = train_samplers['mix']

        scheduler_model = torch.optim.lr_scheduler.MultiStepLR(optimizer_score,config.optim.lr_step_model, config.optim.lr_factor_model,last_epoch= start_epoch-1)
        scheduler_model_cond = torch.optim.lr_scheduler.MultiStepLR(optimizer_score_cond, config.optim.lr_step_hrnet, config.optim.lr_factor_hrnet,last_epoch= start_epoch-1)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            print('lr lcn:{} lr hrnet:{}'.format(optimizer_score.state_dict()['param_groups'][0]['lr'],optimizer_score_cond.state_dict()['param_groups'][0]['lr']))
            loss_total= []
            loss_pve =[]
            loss_mcam = []
            loss_2d = []
            train_sampler.set_epoch(epoch)
            train_loader1 = tqdm(train_loader, dynamic_ncols=True)
            for i,  (inps, labels, img_ids, bboxes) in enumerate(train_loader1):
                optimizer_score.zero_grad()
                optimizer_score_cond.zero_grad()
                
                step += 1
                multi_n = np.array(config.training.scorenet.cases).sum()
                
                n = inps.size(0)
                
                for k, _ in labels.items():
                    labels[k] = labels[k].to(self.device)
                input = inps.float().to(self.device)
                joints = labels['joints_uvd_29'][:,:self.num_joints].float().to(self.device)
                twist =labels['target_twist'].float().to(self.device).view(n,-1)
                scale = torch.tensor([self.image_size[0],self.image_size[1],train_dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                labels['trans_inv']= labels['trans_inv'].unsqueeze(1).repeat(1,multi_n,1,1).view(-1,2,3)
                labels['intrinsic_param']= labels['intrinsic_param'].unsqueeze(1).repeat(1,multi_n,1,1).view(-1,3,3)
                labels['joint_root']= labels['joint_root'].unsqueeze(1).repeat(1,multi_n,1).view(-1,3)
                
                ''' generation process '''
                output = {}
                score_input_list = {'joint':[], 'twist':[]}
                gen_output_list = {'pred_joints':[], 'pred_twist':[]}
                shuffle_idx = torch.randperm(multi_n)
                for k in range(num_models):
                    state = {
                        'model_cond': model_cond[k],
                        'model': model[k],
                        'input': input.clone(),
                        'scale': scale.clone(),
                    }

                    gen_output, score_input = self.gen_mesh(state, config.training.scorenet.cases[k])
                    for key in gen_output_list.keys():
                        gen_output_list[key].append(gen_output[key])
                    for key in score_input_list.keys():
                        score_input_list[key].append(score_input[key])
                for key in gen_output_list:
                    gen_output_list[key] = torch.cat(gen_output_list[key],dim=1)
                    gen_output_list[key] = gen_output_list[key][:, shuffle_idx]
                for key in score_input_list:
                    score_input_list[key] = torch.cat(score_input_list[key],dim=1)
                    score_input_list[key] = score_input_list[key][:, shuffle_idx]
                    score_input_list[key] = score_input_list[key].view(n*multi_n, -1)

                output['pred_twist'] = gen_output_list['pred_twist'].view(-1,self.num_twists,2)
                output['pred_joints'] = gen_output_list['pred_joints'].view(-1,self.num_joints,3)
                output['pred_shape'] = labels['target_beta'].unsqueeze(1).repeat(1,multi_n,1).view(-1,10)
                output_final = process_output(output.copy(), labels.copy(), self.smpl)
                
                cond_feature, output_final['pred_2d'] = model_score_cond(input.clone())
                joint_2d = labels['joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device)
                scale_2d = torch.tensor([self.image_size[0],self.image_size[1]]).float().to(self.device)
                joint_2d = normalize_pose_cuda(pose = joint_2d.clone(), which='scale_t',scale=scale_2d).to(self.device)
                
                score_input = torch.cat([score_input_list['joint'], score_input_list['twist']],dim=1).to(self.device)
                score = model_score(score_input, cond_feature)
                
                
                gt_betas = labels['target_beta']
                gt_thetas = labels['target_theta']
                gt_output = self.smpl(
                    pose_axis_angle=gt_thetas,
                    betas=gt_betas,
                    global_orient=None,
                    return_verts=True
                )
                gt_mesh = gt_output.vertices.float()-gt_output.joints_from_verts.float()[:,0].unsqueeze(1)
                gt_mesh = gt_mesh.reshape(n, -1, 3)
                gt = {'mesh': gt_mesh,              # [bs, 6890, 3]
                      'joint_cam': labels['joints_xyz_17'].clone().float().to(self.device),
                      'mask_2d':labels['joints_vis_29'][:,:,:2].clone().float().to(self.device),
                      'pred_2d':joint_2d.clone()}
                
                mask = labels['prelate_weight'].view(-1).to(score.device)
                p_pve, p_mpjpe_cam, loss_2d_cur = loss_score(score, output_final, gt, mask)
                loss = p_pve + p_mpjpe_cam + loss_2d_cur

                loss.backward()
                
                try:
                    torch.nn.utils.clip_grad_norm_(model_score.parameters(), config.optim.grad_clip)
                    torch.nn.utils.clip_grad_norm_(model_score_cond.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                
                
                optimizer_score_cond.step()
                optimizer_score.step()

                loss_total.append(loss.item())
                loss_pve.append(p_pve.item())
                loss_mcam.append(p_mpjpe_cam.item())
                loss_2d.append(loss_2d_cur.item())

                
                if step % self.config.training.loss_freq == 0 or step == 1:
                    if rank ==0:
                        logging.info(f"epoch:{epoch},step: {step}, loss: {loss.item()}, batch size: {n}")
                        logging.info(f"epoch:{epoch},step: {step}, loss pve: {p_pve.item()},  loss mpjpe_cam: {p_mpjpe_cam.item()}, loss 2d:{loss_2d_cur.item()}, batch size: {n}")


                ema_score.update(model_score.parameters())
                ema_score_cond.update(model_score_cond.parameters())


            states = {
                'model_score': model_score.module.state_dict(),
                'optimizer_score': optimizer_score.state_dict(),
                'model_score_cond': model_score_cond.module.state_dict(),
                'optimizer_score_cond': optimizer_score_cond.state_dict(),
                'epoch': epoch,
                'step': step,
                'ema_score': ema_score.state_dict(),
                'ema_score_cond': ema_score_cond.state_dict()
            }
            if rank==0:
                torch.save(states, os.path.join(self.config.log_path, 'ckpt_epoch_{}.pth'.format(epoch)))
                torch.save(states, os.path.join(self.config.log_path, 'ckpt.pth'.format(epoch)))
                loss_total = np.array(loss_total).mean()
                loss_pve = np.array(loss_pve).mean()
                loss_mcam = np.array(loss_mcam).mean()
                loss_2d = np.array(loss_2d).mean()
                logger.info(f"epoch:{epoch},train_loss: {loss_total}, pve:{loss_pve} mcam:{loss_mcam} joint_2d:{loss_2d}")
            train_loader1.close()
            scheduler_model.step()
            scheduler_model_cond.step()
            
            if epoch % self.config.training.validation_freq == 0:
                torch.distributed.barrier()
                ema_score.store(model_score.parameters())
                ema_score.copy_to(model_score.parameters())
                ema_score_cond.store(model_score_cond.parameters())
                ema_score_cond.copy_to(model_score_cond.parameters())
                model_score.eval()
                model_score_cond.eval()
                states = {
                    'model_score': model_score,
                    'model_score_cond': model_score_cond,
                    'epoch': epoch
                }
                error_dict = self.validate(states)
                ema_score.restore(model_score.parameters())
                ema_score_cond.restore(model_score_cond.parameters())
                model_score.train()
                model_score_cond.train()
            torch.distributed.barrier()

    def validate(self ,state=None):
        args, config = self.args, self.config
        
        rank = torch.distributed.get_rank()
        model, model_cond, __, __, __, __, __, __, __, __, __ = get_model(config, is_train=False, resume = True, resume_path = config.training.scorenet.test_path)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)
        model_cond = nn.parallel.DistributedDataParallel(model_cond, device_ids=[args.device], output_device=args.local_rank)


        if state==None:
            if os.path.exists(getattr(self.config.sampling, "ckpt", None)):
                resume_path = self.config.sampling.ckpt
            elif getattr(self.config.sampling, "ckpt", None) is None:
                resume_path = os.path.join(self.config.log_path, "ckpt.pth")
            else:
                resume_path = os.path.join(self.config.log_path, self.config.sampling.ckpt)
            
            model_score, model_score_cond, __, __, __, __, epoch, step, __ = get_model_score(config, is_train=False, resume = True, resume_path = resume_path)
            model_score = nn.parallel.DistributedDataParallel(model_score, device_ids=[args.device], output_device=args.local_rank)
            model_score_cond = nn.parallel.DistributedDataParallel(model_score_cond, device_ids=[args.device], output_device=args.local_rank)
            state = dict(model_score=model_score,epoch=epoch,model_score_cond=model_score_cond)
        
        epoch = state['epoch']
        state['model'] = model
        state['model_cond'] = model_cond
        
        valid_loaders, valid_datasets, __ = get_dataloader(config, is_train = False)
        
        pred = {}
        for test_dataset in self.config.dataset.test_dataset:
            state['dataset'] = valid_datasets[test_dataset]
            state['dataloader'] =  valid_loaders[test_dataset]
            state['dataset_type'] = test_dataset
            pred[test_dataset] = self.sample(state, args.construct_dpo)
            torch.distributed.barrier()
        return pred
    

    def sample(self, state, construct_dpo=False):
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
            dpo_pair_list = []
            for i in range(multi_n+2):
                pred[i] = {}
            for i, (inps, labels, img_ids, bboxes, img_path) in enumerate(tqdm(dataloader, ncols=100)):
                n = inps.size(0)
                output = {}
                for k, _ in labels.items():
                    labels[k] = labels[k].to(self.device)
                input = inps.float().to(self.device)
                save_trans_inv = labels['trans_inv'].cpu().numpy().astype(np.float32)
                scale = torch.tensor([self.image_size[0],self.image_size[1],dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                labels['trans_inv']= labels['trans_inv'].unsqueeze(1).repeat(1,multi_n+2,1,1).view(-1,2,3)
                labels['intrinsic_param']= labels['intrinsic_param'].unsqueeze(1).repeat(1,multi_n+2,1,1).view(-1,3,3)
                labels['joint_root']= labels['joint_root'].unsqueeze(1).repeat(1,multi_n+2,1).view(-1,3)

                state['input'] = input.clone()
                state['scale'] = scale.clone()
                output, score_input = self.gen_mesh(state, multi_n)                     # get denoised twist and joints

                if construct_dpo:
                    # construct with GT as winner
                    save_trans_inv = save_trans_inv.tolist()
                    save_pred_joints = output['pred_joints'].cpu().numpy().astype(np.float32).tolist()
                    save_pred_twist = output['pred_twist'].cpu().numpy().astype(np.float32).tolist()
                    save_score_joint = score_input['joint'].cpu().numpy().astype(np.float32).tolist()
                    save_score_twist = score_input['twist'].cpu().numpy().astype(np.float32).tolist()
                    save_joints_uvd_29 = labels['joints_uvd_29'].cpu().numpy().astype(np.float32).tolist()
                    save_w_twist = labels['target_twist'].cpu().numpy().astype(np.float32).tolist()
                    save_trans = labels['trans'].cpu().numpy().astype(np.float32).tolist()
                    # import ipdb; ipdb.set_trace()
                    for idx in range(n):
                        for idx_hypo in range(multi_n):
                            dpo_pair = {
                                'img_path': img_path[idx],
                                # 'db_idx': labels['db_idx'][idx].item(),
                                'img_idx': img_ids[idx].item(),                     # absolute img_ids 
                                'l_joints': save_pred_joints[idx][idx_hypo],
                                'l_twist': save_pred_twist[idx][idx_hypo],
                                'w_joints': save_joints_uvd_29[idx],
                                'w_twist': save_w_twist[idx],
                                'trans': save_trans[idx],
                                'trans_inv': save_trans_inv[idx],
                            }
                            dpo_pair_list.append(dpo_pair)
                    continue


                output['pred_shape'] = output['pred_shape'].unsqueeze(1).repeat(1, multi_n+2, 1).view(-1,10)
                output['pred_joints'] = output['pred_joints'].view(-1,self.num_joints,3)        # 1600, 29, 3
                output['pred_twist'] = output['pred_twist'].view(-1,self.num_twists,2)          # 1600, 23, 2

                
                
                cond_feature,__ = model_score_cond(input.clone())
                
                score_input = torch.cat([score_input['joint'].view(n*multi_n,-1),score_input['twist'].view(n*multi_n,-1)],dim=1).to(self.device)
                score = model_score(score_input, cond_feature)

                
                score = score.view(n, multi_n)
                joints = output['pred_joints'].view(n,multi_n,self.num_joints,3)
                twist = output['pred_twist'].view(n,multi_n,self.num_twists,2)
                idx_score = score.contiguous().argsort(dim=1)[:, -self.topk:].contiguous().view(-1)
                idx_bs = torch.arange(n).repeat(self.topk).sort()[0]
                
                select_score = score[idx_bs, idx_score].view(n, self.topk)
                select_score = torch.nn.functional.softmax(select_score,dim=-1).unsqueeze(-1)


                select_joints = joints[idx_bs, idx_score].view(n,self.topk,self.num_joints,3).mean(dim=1)           # take the average of topk predictions 160, 29, 3
                #select_joints = (select_joints*select_score.unsqueeze(-1)).sum(dim=1)


                select_twist = twist[idx_bs,idx_score].view(n,self.topk,self.num_twists,2)
                select_twist = select_twist/(torch.norm(select_twist, dim=-1, keepdim=True) + 1e-8)
                select_angle = torch.arctan(select_twist[:,:,:,1]/select_twist[:,:,:,0]) 
                flag = (torch.cos(select_angle)*select_twist[:,:,:,0])<0
                select_angle = select_angle + flag*np.pi
                assert ((torch.cos(select_angle)-select_twist[:,:,:,0]).abs()>1e-6).sum() ==0
                assert ((torch.sin(select_angle)-select_twist[:,:,:,1]).abs()>1e-6).sum() ==0
                select_angle = select_angle.mean(dim=1)                                                             # take the average of topk predictions
                #select_angle = (select_angle*select_score).sum(dim=1)
                select_twist = select_twist.mean(dim=1)
                select_twist = torch.cat([torch.cos(select_angle).unsqueeze(-1), torch.sin(select_angle).unsqueeze(-1)], dim=-1)

                output['pred_joints'] = output['pred_joints'].view(n, multi_n, self.num_joints, 3)
                mean_joint = output['pred_joints'].mean(dim=1).unsqueeze(1)
                output['pred_joints'] = torch.cat([select_joints.unsqueeze(1), output['pred_joints'], mean_joint], dim=1).view(-1, self.num_joints, 3)      # [bs x (topk_mean + multi_n + all hypos' mean), 29, 3]
                output['pred_twist'] = output['pred_twist'].view(n, multi_n, self.num_twists, 2)
                mean_twist = output['pred_twist'].mean(dim=1).unsqueeze(1)
                output['pred_twist'] = torch.cat([select_twist.unsqueeze(1),output['pred_twist'], mean_twist],dim=1).view(-1,self.num_twists,2)      # [bs x (topk_mean + multi_n + all hypos' mean), 23, 2]

                score = torch.cat([torch.zeros(n,1).to(score.device) + score.max()+50, score.view(n, multi_n), torch.zeros(n,1).to(score.device)],dim=-1)
                score = score.view(n, multi_n+2).cpu().data.numpy()
            
                pred_shape = output['pred_shape'].clone().cpu().data.numpy()
                pred_uvd_jts_29 = trans_back(output['pred_joints'].clone(), labels['trans_inv'].clone()).cpu().data.numpy()
                pred_twist = output['pred_twist'].clone().cpu().data.numpy()
                
                pred_shape = pred_shape.reshape(n, multi_n+2, 10)
                pred_twist = pred_twist.reshape(n, multi_n+2, 23, 2)
                pred_uvd_jts_29 = pred_uvd_jts_29.reshape(n, multi_n+2, 29, 3)

                output_final = process_output(output, labels, self.smpl)
                pred_joints = output_final['pred_joints'].clone().cpu().data.numpy() * 2
                pred_joints = pred_joints.reshape(n, multi_n+2, 29, 3)

                gt_betas = labels['target_beta'].view(-1, 10)
                gt_thetas = labels['target_theta'].view(-1, 96)
                gt_output = self.smpl(
                        pose_axis_angle=gt_thetas,
                        betas=gt_betas,
                        global_orient=None,
                        return_verts=True
                    )
                # vertice
                pred_mesh = output_final['pred_vertices'].reshape(n, multi_n+2, -1, 3)
                draw_mesh = pred_mesh.view(n,multi_n+2,-1,3).cpu().data.numpy()
                gt_mesh = gt_output.vertices.float()-gt_output.joints_from_verts.float()[:,0].unsqueeze(1)
                gt_mesh = gt_mesh.reshape(n,1,-1,3)
                pred_mesh = pred_mesh.cpu().data.numpy()
                gt_mesh = gt_mesh.cpu().data.numpy()
                pred_xyz_jts_17 = output_final['pred_xyz_jts_17'].reshape(n, multi_n+2, 17, 3).cpu().data.numpy()
                pred_theta = output_final['theta'].reshape(n, multi_n+2 ,24, 4).cpu().data.numpy()
                '''
                if self.config.write_obj:
                    path_s = os.path.join(config.save_path,str(self.multihypo_n),'mesh',state['dataset_type'])
                    for k in range(multi_n//2+1):
                        for j in range(pred_xyz_jts_17.shape[0]):
                            path_i = os.path.join(path_s,str(int(img_ids[j])))
                            os.makedirs(path_i,exist_ok=True)
                            path_i = os.path.join(path_i,str(k)+'.obj')
                            write_obj(path_i,draw_mesh[j][k],self.smpl.faces)
                '''
                
                pve = np.sqrt(np.sum((pred_mesh - gt_mesh) ** 2, axis=-1))
                pve = np.mean(pve,axis=-1) * 1000
                pve = pve.reshape(n, multi_n+2)
                
                for k in range(multi_n+2):
                    for j in range(pred_xyz_jts_17.shape[0]):
                        item_i = {'xyz_17': pred_xyz_jts_17[j][k],
                                'score': score[j][k],
                                'uvd_29': pred_uvd_jts_29[j][k],
                                'shape': pred_shape[j][k],
                                'twist': pred_twist[j][k],
                                'pred': pred_joints[j][k],
                                'pve': pve[j][k],
                                'theta': pred_theta[j][k]
                                }
                        pred[k][int(img_ids[j])] = item_i
        
        if construct_dpo:
            # save_path = f'data/dpo/annotations/new_{dataset.__class__.__name__}_train_interval_{config.sampling.interval}_multihypo_{config.sampling.multihypo_n}.npz'
            # if os.path.exists(save_path):
                # print(f"Overwrite {save_path} !!!")
            # Cache.save(save_path, dpo_pair_list)
            save_path = f'data/dpo/annotations/gt_sorted_{dataset.__class__.__name__}_train_interval_{config.sampling.interval}_multihypo_{config.sampling.multihypo_n}.json'
            if os.path.exists(save_path):
                print(f"Overwrite {save_path} !!!")
            print(f"DPO pair list length: {len(dpo_pair_list)}")
            with open(save_path, 'w') as json_file:
                json.dump(dpo_pair_list, json_file)
            print(f'[{dataset.__class__.__name__}] Caching datalist to {save_path}...')
            print(f"Saving DPO data pairs to {save_path}")
            sys.exit(0)

        name = 'validate' + str(epoch)
        save_path=os.path.join(self.config.save_path, name)
        path = save_path+'_'+str(rank)+'.pkl'
        with open(path, 'wb') as fid:
            pickle.dump(pred, fid, pickle.HIGHEST_PROTOCOL)
        print('dump the file', path)
        torch.distributed.barrier()
        name = 'validate' + str(epoch)
        save_path = os.path.join(self.config.save_path, name)
        if rank==0:
            print('gpu num: {}'.format(args.world_size))
            pred = {}
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
            save_path= self.config.save_path
            p_relate = process_pred(pred, dataset, multi_n+2, type=dataset_type, save_path=save_path, use_score=True)
        else:
            p_relate = {}
        torch.distributed.barrier()
        return p_relate
        

    def sample_pose(self, xj, xt, model,content=None, last=True,gen_multi=True):
        
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

        xs = generalized_steps(xj, xt, seq, model, self.betas.to(self.device), eta=self.config.diffusion.eta,ctx =content, gen_multi=gen_multi)
        x = xs
        if last:                                    # get the predicted x_0, abandon intermediate results
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
        gen_output['pred_shape'] = pred_shape                                   # bs, 10

        n = input.shape[0]
        if self.config.sampling.zero and self.config.sampling.multihypo_n==1:
            xj = torch.zeros(n*multi_n, self.num_joints*3).to(self.device)
            xt = torch.zeros(n*multi_n, self.num_twists*2).to(self.device)
        else:
            xj = torch.randn(n,multi_n, self.num_joints*3).to(self.device)      # swing represenataion in joint positions [bs, multi_n. n_joints x 3]
            xt = torch.randn(n,multi_n, self.num_twists*2).to(self.device)      # twist representation
            xj[:,0] = 0
            xj = xj.view(n*multi_n, self.num_joints*3).to(self.device)
            xt[:,0] = 0
            xt = xt.view(n*multi_n, self.num_twists*2).to(self.device)
        
        pred_j, pred_t= self.sample_pose(xj=xj, xt=xt, model=model, content=ctx, gen_multi=True)
        gen_output['pred_joints'] = denormalize_pose_cuda(pose = pred_j, which=self.config.hyponet.norm, scale=scale, \
                                                          mean_and_std=self.config.diffusion.scale).to(self.device).view(n,multi_n,self.num_joints,3)       # bs, multi_n, 29, 3
        gen_output['pred_twist'] = pred_t.view(n, multi_n,self.num_twists,2) / self.config.diffusion.scale                                                  # bs, multi_n, 23, 2

        score_input['joint'] = normalize_pose_cuda(pose = gen_output['pred_joints'].clone().view(-1, self.num_joints, 3), \
                                                   which='scale_t', scale=scale*self.config.diffusion.scale).to(self.device).view(n,multi_n,self.num_joints*3)
        score_input['twist'] = gen_output['pred_twist'].clone()

        return gen_output, score_input