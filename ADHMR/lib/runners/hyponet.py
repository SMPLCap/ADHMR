import os
import logging
import time
import glob
from tqdm import tqdm
import pickle

import numpy as np
import torch
from collections import defaultdict
from utils.pose_utils import *
from utils.diff_utils import * 
from utils.function import get_optimizer, get_model, get_dataloader, process_pred
from utils.relation import *
from models.layers.smpl.SMPL import SMPL_layer
import random

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class HypoNetTrainer(object):
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
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alpha = alphas_cumprod
        h36m_jregressor = np.load('data/smpl/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            'data/smpl/SMPL_NEUTRAL.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        ).to(self.device)
        self.rank =  torch.distributed.get_rank()
        print('rank {} get train now! '.format(self.rank))

    def train(self):
        args, config = self.args, self.config
        
        if config.training.get('dpo', False) or config.training.get('kto', False):
            model, model_cond, ref_model, ref_model_cond, ema_helper, ema_helper_cond, optimizer_hyponet, optimizer_hrnet, start_epoch, step, loss_smpl, min_mpjpe_h36m, min_mpjpe_pw3d = \
                get_model(config, is_train=True, resume = config.training.resume_training, resume_path = config.training.resume_ckpt)
        else:
            model, model_cond, ema_helper, ema_helper_cond, optimizer_hyponet, optimizer_hrnet, start_epoch, step, loss_smpl, min_mpjpe_h36m, min_mpjpe_pw3d = \
                get_model(config, is_train=True, resume = config.training.resume_training, resume_path = config.training.resume_ckpt)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank, find_unused_parameters=True)
        model_cond = nn.parallel.DistributedDataParallel(model_cond, device_ids=[args.device], output_device=args.local_rank, find_unused_parameters=True)

        last_epoch = start_epoch - 1
        scheduler_model = torch.optim.lr_scheduler.MultiStepLR(optimizer_hyponet, config.optim.lr_step_model, config.optim.lr_factor_model, last_epoch=last_epoch)
        scheduler_model_cond = torch.optim.lr_scheduler.MultiStepLR(optimizer_hrnet, config.optim.lr_step_hrnet, config.optim.lr_factor_hrnet,last_epoch=last_epoch)


        train_loaders, train_datasets, train_samplers = get_dataloader(config, is_train = True)
        # import ipdb; ipdb.set_trace()
        if config.training.get('dpo', False):
            train_loader = train_loaders['dpo']
            train_dataset = train_datasets['dpo']
            train_sampler = train_samplers['dpo']
        elif config.training.get('kto', False):
            train_loader = train_loaders['kto']
            train_dataset = train_datasets['kto']
            train_sampler = train_samplers['kto']
        else:
            train_loader = train_loaders['mix']
            train_dataset = train_datasets['mix']
            train_sampler = train_samplers['mix']
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            print('lr lcn:{} lr hrnet:{}'.format(optimizer_hyponet.state_dict()['param_groups'][0]['lr'],optimizer_hrnet.state_dict()['param_groups'][0]['lr']))
            model.train()
            model_cond.train()
            if config.training.get('dpo', False):
                model_cond.eval()
                ref_model_cond.eval()
                ref_model.eval()
            if config.training.finetune:
                model_cond.eval()
            loss_total= []
            train_sampler.set_epoch(epoch)
            if self.rank==0:
                train_loader1 = tqdm(train_loader, dynamic_ncols=True)
            else:
                train_loader1 = train_loader
            for i,  (inps, labels, img_ids, bboxes) in enumerate(train_loader1):
                optimizer_hyponet.zero_grad()
                optimizer_hrnet.zero_grad()
                n = inps.size(0)
                for k, _ in labels.items():
                    labels[k] = labels[k].to(self.device)
                input = inps.float().to(self.device)
                if config.training.get('dpo', False):
                    twist = labels['target_twist'].clone().float().to(self.device).view(n,-1) * self.config.diffusion.scale
                    l_twist = labels['target_l_twist'].clone().float().to(self.device).view(n,-1) * self.config.diffusion.scale
                    w_twist = labels['target_w_twist'].clone().float().to(self.device).view(n,-1) * self.config.diffusion.scale
                    twist_wl = torch.cat([w_twist, l_twist], dim=0)


                    joints = labels['joints_uvd_29'][:,:self.num_joints].float().to(self.device).clone()
                    joint_2d = labels['joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device).clone()
                    scale_2d = torch.tensor([self.image_size[0],self.image_size[1]]).float().to(self.device)
                    joint_2d = normalize_pose_cuda(pose = joint_2d.clone(), which='scale_t', scale=scale_2d).to(self.device)
                    scale = torch.tensor([self.image_size[0], self.image_size[1], train_dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                    x = normalize_pose_cuda(pose = joints, which=config.hyponet.norm, scale=scale, mean_and_std=self.config.diffusion.scale).to(self.device)

                    l_joints = labels['l_joints_uvd_29'][:,:self.num_joints].float().to(self.device).clone()
                    l_joint_2d = labels['l_joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device).clone()
                    l_joint_2d = normalize_pose_cuda(pose = l_joint_2d.clone(), which='scale_t', scale=scale_2d).to(self.device)
                    l_x = normalize_pose_cuda(pose = l_joints, which=config.hyponet.norm, scale=scale, mean_and_std=self.config.diffusion.scale).to(self.device)
                    
                    w_joints = labels['w_joints_uvd_29'][:,:self.num_joints].float().to(self.device).clone()
                    w_joint_2d = labels['w_joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device).clone()
                    w_joint_2d = normalize_pose_cuda(pose = w_joint_2d.clone(), which='scale_t', scale=scale_2d).to(self.device)
                    w_x = normalize_pose_cuda(pose = w_joints, which=config.hyponet.norm, scale=scale, mean_and_std=self.config.diffusion.scale).to(self.device)

                    x_wl = torch.cat([w_x, l_x], dim=0)

                    ctx, pred_shape, pred_2d = model_cond(input)
                    ctx['global'] = ctx['global'].repeat(2, 1, 1, 1)
                    ctx['local'] = ctx['local'].repeat(2, 1, 1)
                    ctx_ref, pred_shape_ref, pred_2d_ref = ref_model_cond(input)
                    ctx_ref['global'] = ctx_ref['global'].repeat(2, 1, 1, 1)
                    ctx_ref['local'] = ctx_ref['local'].repeat(2, 1, 1)

                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    t = t.repeat(2)
                    ej = torch.randn_like(x).to(self.device)
                    ej = ej.repeat(2, 1)
                    et = torch.randn_like(twist).to(self.device)
                    et = et.repeat(2, 1)
                    gt = {'noise_j': ej, 'noise_t': et, 'joint_2d': joint_2d.view(-1, self.num_joints, 2)}
                    a = self.alpha.clone().to(self.device).index_select(0, t).view(-1, 1)
                    
                    # Forward Process
                    xinj = x_wl * a.sqrt() + ej * (1.0 - a).sqrt()
                    xint = twist_wl * a.sqrt() + et * (1.0 - a).sqrt()
                    
                    # Reverse Process
                    output_list = model(xinj=xinj, xint=xint, t=t.float(), ctx=ctx)
                    output = {'pred_shape': pred_shape, 'joint_2d': pred_2d.view(-1, self.num_joints, 2), 'noise_j': output_list[0], 'noise_t': output_list[1]}
                    loss_origin, loss_list = loss_smpl(output, gt, labels)
                    model_joint_diff = loss_list['joint_diff']
                    model_twist_diff = loss_list['twist_diff']

                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        ref_output_list = ref_model(xinj=xinj, xint=xint, t=t.float(), ctx=ctx_ref)
                        ref_output = {'pred_shape': pred_shape_ref, 'joint_2d': pred_2d_ref.view(-1, self.num_joints, 2), 'noise_j': ref_output_list[0].detach(), 'noise_t': ref_output_list[1].detach()}
                        ref_loss, ref_loss_list = loss_smpl(ref_output, gt, labels)
                        ref_joint_diff = ref_loss_list['joint_diff']
                        ref_twist_diff = ref_loss_list['twist_diff']
                    
                    # DPO loss
                    scale_term = -0.5 * args.beta_dpo
                    joint_inside_term = scale_term * (model_joint_diff - ref_joint_diff)
                    # joint_implicit_acc = (joint_inside_term > 0).sum().float() / joint_inside_term.size(0)
                    dpo_joint_loss = -1 * F.logsigmoid(joint_inside_term).mean()

                    twist_inside_term = scale_term * (model_twist_diff - ref_twist_diff)
                    # twist_implicit_acc = (twist_inside_term > 0).sum().float() / twist_inside_term.size(0)
                    dpo_twist_loss = -1 * F.logsigmoid(twist_inside_term).mean()

                    loss = dpo_joint_loss + dpo_twist_loss

                elif config.training.get('kto', False):
                    joints = labels['joints_uvd_29'][:,:self.num_joints].float().to(self.device)
                    twist = labels['target_twist'].float().to(self.device).view(n,-1) * self.config.diffusion.scale
                    joint_2d = labels['joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device)
                    scale_2d = torch.tensor([self.image_size[0],self.image_size[1]]).float().to(self.device)
                    joint_2d = normalize_pose_cuda(pose = joint_2d.clone(), which='scale_t', scale=scale_2d).to(self.device)
                    scale = torch.tensor([self.image_size[0], self.image_size[1], train_dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                    x = normalize_pose_cuda(pose = joints, which=config.hyponet.norm, scale=scale, mean_and_std=self.config.diffusion.scale).to(self.device)
                    
                    ctx, pred_shape, pred_2d = model_cond(input)
                    ctx_ref, pred_shape_ref, pred_2d_ref = ref_model_cond(input)

                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    ej = torch.randn_like(x).to(self.device)
                    et = torch.randn_like(twist).to(self.device)
                    gt = {'noise_j': ej, 'noise_t': et, 'joint_2d': joint_2d.view(-1, self.num_joints, 2)}
                    a = self.alpha.clone().to(self.device).index_select(0, t).view(-1, 1)
                    
                    # Forward Process
                    xinj = x * a.sqrt() + ej * (1.0 - a).sqrt()
                    xint = twist * a.sqrt() + et * (1.0 - a).sqrt()
                    # Reverse Process                                        
                    output_list = model(xinj=xinj, xint=xint, t=t.float(), ctx=ctx)
                    output = {'pred_shape': pred_shape, 'joint_2d': pred_2d.view(-1, self.num_joints, 2), 'noise_j': output_list[0], 'noise_t': output_list[1]}
                    loss_origin, loss_list = loss_smpl(output, gt, labels)
                    model_loss_joint = loss_list['loss_joint']
                    model_loss_twist = loss_list['loss_twist']

                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        ref_output_list = ref_model(xinj=xinj, xint=xint, t=t.float(), ctx=ctx_ref)
                        ref_output = {'pred_shape': pred_shape_ref, 'joint_2d': pred_2d_ref.view(-1, self.num_joints, 2), 'noise_j': ref_output_list[0].detach(), 'noise_t': ref_output_list[1].detach()}
                        ref_loss, ref_loss_list = loss_smpl(ref_output, gt, labels)
                        ref_loss_joint = ref_loss_list['loss_joint']
                        ref_loss_twist = ref_loss_list['loss_twist']
                    
                    # KTO loss
                    scale_term = args.beta_dpo
                    labels = labels["label_sgn"]
                    label_sgn = 2 * labels - 1
                    labels_binary = labels == 1
                    w_y = (
                        args.lambda_d_kto * labels_binary
                        + args.lambda_u_kto * ~labels_binary
                    )

                    # Final loss.
                    policy_KL_logps = - model_loss_joint
                    reference_KL_logps = - ref_loss_joint
                    g_term_joint = policy_KL_logps - reference_KL_logps
                    kl = g_term_joint.mean().clamp(min=0).detach()
                    g_term_joint = g_term_joint - kl
                    label_scale_g = label_sgn * scale_term * g_term_joint
                    h = torch.sigmoid(label_scale_g)
                    l_kto_joint = w_y * (1 - h)
                    l_kto_joint = l_kto_joint.mean()

                    policy_KL_logps = - model_loss_twist
                    reference_KL_logps = - ref_loss_twist
                    g_term_twist = policy_KL_logps - reference_KL_logps
                    kl = g_term_twist.mean().clamp(min=0).detach()
                    g_term_twist = g_term_twist - kl
                    label_scale_g = label_sgn * scale_term * g_term_twist
                    h = torch.sigmoid(label_scale_g)
                    l_kto_twist = w_y * (1 - h)
                    l_kto_twist = l_kto_twist.mean()

                    loss = l_kto_joint + l_kto_twist + loss_origin * 0.1
                else:
                    joints = labels['joints_uvd_29'][:,:self.num_joints].float().to(self.device)
                    twist = labels['target_twist'].float().to(self.device).view(n,-1) * self.config.diffusion.scale
                    joint_2d = labels['joints_uvd_29'][:,:self.num_joints,:2].float().to(self.device)
                    scale_2d = torch.tensor([self.image_size[0],self.image_size[1]]).float().to(self.device)
                    joint_2d = normalize_pose_cuda(pose = joint_2d.clone(), which='scale_t', scale=scale_2d).to(self.device)
                    scale = torch.tensor([self.image_size[0], self.image_size[1], train_dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                    x = normalize_pose_cuda(pose = joints, which=config.hyponet.norm, scale=scale, mean_and_std=self.config.diffusion.scale).to(self.device)
                    
                    ctx, pred_shape, pred_2d = model_cond(input)

                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    ej = torch.randn_like(x).to(self.device)
                    et = torch.randn_like(twist).to(self.device)
                    gt = {'noise_j': ej, 'noise_t': et, 'joint_2d': joint_2d.view(-1, self.num_joints, 2)}
                    a = self.alpha.clone().to(self.device).index_select(0, t).view(-1, 1)
                    
                    # Forward Process
                    xinj = x * a.sqrt() + ej * (1.0 - a).sqrt()
                    xint = twist * a.sqrt() + et * (1.0 - a).sqrt()
                    # Reverse Process
                    output_list = model(xinj=xinj, xint=xint, t=t.float(), ctx=ctx)
                    output = {'pred_shape': pred_shape, 'joint_2d': pred_2d.view(-1, self.num_joints, 2), 'noise_j': output_list[0], 'noise_t': output_list[1]}
                    loss, loss_list = loss_smpl(output, gt, labels)


                # loss and ema update
                loss.backward()
                
                # clip the grad
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                    torch.nn.utils.clip_grad_norm_(model_cond.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                
                optimizer_hrnet.step()
                optimizer_hyponet.step()

                ema_helper.update(model.parameters())
                ema_helper_cond.update(model_cond.parameters())

                # loss log and collect
                loss_total.append(loss.item())
                if step % self.config.training.loss_freq == 0 and self.rank ==0:
                    if config.training.get('dpo', False):
                        logging.info(f"epoch:{epoch}, step: {step}, loss: {loss.item()}, batch size: {n}, loss_origin: {loss_origin.item()} \
                                 loss_joint_w_mean: {loss_list['losses_joint_w'].mean()}, loss_joint_l_mean: {loss_list['losses_joint_l'].mean()}, \
                                 loss_twist_w_mean: {loss_list['losses_twist_w'].mean()}, loss_twist_l_mean: {loss_list['losses_twist_l'].mean()}")
                    elif config.training.get('kto', False):
                        logging.info(f"epoch:{epoch}, step: {step}, loss: {loss.item()}, batch size: {n}, loss_origin: {loss_origin.item()}, l_kto_joint: {l_kto_joint}, l_kto_twist: {l_kto_twist}")
                    else:
                        logging.info(f"epoch:{epoch}, step: {step}, loss: {loss.item()}, batch size: {n}")
                step += 1
            
            if self.rank==0:
                states = {
                    'optimizer': optimizer_hyponet.state_dict(),
                    'optimizer_cond': optimizer_hrnet.state_dict(),
                    'ema': ema_helper.state_dict(),
                    'ema_cond': ema_helper_cond.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'min_mpjpe_h36m': min_mpjpe_h36m,
                    'min_mpjpe_pw3d': min_mpjpe_pw3d,
                    'model_cond':  model_cond.module.state_dict(),
                    'model': model.module.state_dict()
                }
                
                torch.save(states, os.path.join(self.config.log_path, 'ckpt_epoch_{}.pth'.format(epoch)))
                torch.save(states, os.path.join(self.config.log_path, 'ckpt.pth'.format(epoch)))
                loss_total = np.array(loss_total).mean()
                logger.info(f"epoch:{epoch},train_loss: {loss_total}")
                train_loader1.close()
            scheduler_model.step()
            scheduler_model_cond.step()

            if epoch % self.config.training.validation_freq == 0:
                torch.distributed.barrier()
                ema_helper.store(model.parameters())
                ema_helper.copy_to(model.parameters())
                ema_helper_cond.store(model_cond.parameters())
                ema_helper_cond.copy_to(model_cond.parameters())
                model.eval()
                model_cond.eval()
                valid_state = {
                    'model':model,
                    'model_cond': model_cond,
                    'epoch': epoch
                }
                metric_dict = self.validate(valid_state)
                ema_helper.restore(model.parameters())
                ema_helper_cond.restore(model_cond.parameters())

                if self.rank == 0:
                    if 'h36m' in self.config.dataset.test_dataset:
                        if metric_dict['h36m']['mpjpe'] < min_mpjpe_h36m and epoch >= config.training.save_best:
                            torch.save(states, os.path.join(self.config.log_path, 'best_h36m.pth'.format(epoch)))
                            min_mpjpe_h36m = metric_dict['h36m']['mpjpe']
                            print(f"best epoch of h36m with epoch {epoch} and mpjpe{ metric_dict['h36m']['mpjpe']:.2f}")
                    if '3dpw' in self.config.dataset.test_dataset:
                        if metric_dict['3dpw']['mpjpe'] < min_mpjpe_pw3d and epoch >= config.training.save_best:
                            torch.save(states, os.path.join(self.config.log_path, 'best_pw3d.pth'.format(epoch)))
                            min_mpjpe_pw3d = metric_dict['3dpw']['mpjpe']
                            print(f"best epoch of 3dpw with epoch {epoch} and mpjpe{ metric_dict['3dpw']['mpjpe']:.2f}")


    def validate(self,state = None):
        args, config = self.args, self.config

        if state == None:
            if getattr(self.config.sampling, "ckpt", None) is None:
                resume_path = os.path.join(self.config.log_path, "ckpt.pth")
            else:
                resume_path = os.path.join(self.config.log_path, self.config.sampling.ckpt)
            model, model_cond, ema_helper, ema_helper_cond, __, __, start_epoch, step, __, __, __ = get_model(config, is_train=False, resume = True, resume_path=resume_path)
            
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)
            model_cond = nn.parallel.DistributedDataParallel(model_cond, device_ids=[args.device], output_device=args.local_rank)
            
            
            state = {
                'model':model,
                'model_cond': model_cond,
                'epoch': start_epoch
            }
            
        else:
            start_epoch = state['epoch']

        valid_loaders, valid_datasets, __ = get_dataloader(config, is_train = False)
        
        pred = {}
        for test_dataset in self.config.dataset.test_dataset:
            state['dataset'] = valid_datasets[test_dataset]
            state['dataloader'] =  valid_loaders[test_dataset]
            state['dataset_type'] = test_dataset
            pred[test_dataset] = self.sample(state)
            torch.distributed.barrier()

        return pred
    

    
    def sample(self, state):
        args, config = self.args, self.config
        multi_n = self.config.sampling.multihypo_n
        model = state['model']
        model_cond = state['model_cond']
        dataset = state['dataset']
        dataloader = state['dataloader']
        epoch = state['epoch']
        dataset_type = state['dataset_type']
        
        with torch.no_grad():
            pred = defaultdict(dict)
            for i, (inps, labels, img_ids, bboxes, _) in enumerate(tqdm(dataloader, ncols=100)):
                n = inps.size(0)
                output = {}
                for k, _ in labels.items():
                    labels[k] = labels[k].to(self.device)
                
                input = inps.float().to(self.device)
                scale = torch.tensor([self.image_size[0],self.image_size[1],dataset.bbox_3d_shape[2]]).float().to(self.device) / self.config.diffusion.scale
                labels['trans_inv']= labels['trans_inv'].unsqueeze(1).repeat(1,multi_n,1,1).view(-1,2,3)
                labels['intrinsic_param']= labels['intrinsic_param'].unsqueeze(1).repeat(1,multi_n,1,1).view(-1,3,3)
                labels['joint_root']= labels['joint_root'].unsqueeze(1).repeat(1,multi_n,1).view(-1,3)

                ctx, pred_shape, pred_2d = model_cond(input)
                output['pred_shape'] = pred_shape.unsqueeze(1).repeat(1,multi_n,1).view(-1,10)
                
                if self.config.sampling.zero and multi_n==1:
                    xj = torch.zeros(n*multi_n, self.num_joints*3).to(self.device)
                    xt = torch.zeros(n*multi_n, self.num_twists*2).to(self.device)
                else:
                    xj = torch.randn(n*multi_n, self.num_joints*3).to(self.device).view(n,multi_n,-1)
                    xt = torch.randn(n*multi_n, self.num_twists*2).to(self.device).view(n,multi_n, -1)
                    xj[:,0] = 0
                    xj = xj.view(n*multi_n, self.num_joints*3).to(self.device)
                    xt[:,0]=0
                    xt = xt.view(n*multi_n, self.num_twists*2).to(self.device)
               
                pred_list = self.sample_pose(xj, xt, model=model, content=ctx, gen_multi=(multi_n>1))
                pred_j= pred_list[0]
                pred_t = pred_list[1]

                pred_shape = output['pred_shape'].clone().cpu().data.numpy()
                output['pred_joints'] = denormalize_pose_cuda(pose = pred_j.clone(), which=config.hyponet.norm,scale=scale,mean_and_std=self.config.diffusion.scale).to(self.device)
                pred_uvd_jts_29 = trans_back(output['pred_joints'].clone(),labels['trans_inv']).view(n,multi_n,self.num_joints,3).cpu().data.numpy()
                output['pred_twist'] = pred_t.view(-1,self.num_twists,2)/self.config.diffusion.scale
                pred_twist = output['pred_twist'].view(n,-1,self.num_twists,2).clone().cpu().data.numpy()
                output_final = process_output(output,labels,self.smpl)
                pred_joints = output_final['pred_joints'].clone().cpu().data.numpy() * 2
                pred_joints = pred_joints.reshape(n,multi_n,29,3)

                gt_betas = labels['target_beta']
                gt_thetas = labels['target_theta']
                gt_output = self.smpl(
                    pose_axis_angle=gt_thetas,
                    betas=gt_betas,
                    global_orient=None,
                    return_verts=True
                )
                # vertice
                pred_mesh = output_final['pred_vertices'].reshape(n,multi_n,-1,3).cpu().data.numpy()
                gt_mesh = gt_output.vertices.float()-gt_output.joints_from_verts.float()[:,0].unsqueeze(1)
                gt_mesh = gt_mesh.reshape(n,1,-1,3).cpu().data.numpy()
                pred_xyz_jts_17 = output_final['pred_xyz_jts_17'].reshape(n,multi_n, 17, 3).cpu().data.numpy()
                pred_theta = output_final['theta'].reshape(n,multi_n,24,4).cpu().data.numpy()

                pve = np.sqrt(np.sum((pred_mesh - gt_mesh) ** 2, axis=-1))
                pve = np.mean(pve, axis=-1)* 1000
                pve = pve.reshape(n,multi_n)
                
                for k in range(multi_n):
                    for j in range(pred_xyz_jts_17.shape[0]):
                        item_i = {'xyz_17': pred_xyz_jts_17[j][k],
                                'score': -1,
                                'uvd_29':pred_uvd_jts_29[j][k],
                                'shape':pred_shape[j],
                                'twist':pred_twist[j][k],
                                'pred':pred_joints[j][k],
                                'pve':pve[j][k],
                                'theta':pred_theta[j][k]
                                }
                        pred[k][int(img_ids[j])] = item_i       

        name = 'validate' + str(epoch)
        save_path = os.path.join(self.config.save_path, name)
        path = save_path+'_'+str(self.rank)+'.pkl'
        with open(path, 'wb') as fid:
            pickle.dump(pred, fid, pickle.HIGHEST_PROTOCOL)
        print('dump the file', path)
        pred = defaultdict(dict)
        torch.distributed.barrier()
        if self.rank==0:
            for r in range(self.args.world_size):
                path = save_path+'_'+str(r)+'.pkl'
                with open(path, 'rb') as fid:
                    pred_i = pickle.load(fid)
                    print('load the file',path)
                for midx in range(multi_n):
                    pred[midx].update(pred_i[midx])
                os.remove(path)
            save_path= self.config.save_path
            pred_output = process_pred(pred, dataset, multi_n, type=dataset_type, save_path=save_path)
        else:
            pred_output = {}

        return pred_output
        

    def sample_pose(self, xj,xt, model, content = None, last = True, gen_multi = False):
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

        xs = generalized_steps(x_0=xj,x_1=xt,seq=seq, model=model, b=self.betas.to(self.device), eta=self.config.diffusion.eta, ctx =content, gen_multi = gen_multi)
        x = xs
        if last:
            x = [x[0][0][-1],x[0][1][-1]]
        return x