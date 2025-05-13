import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import pickle, h5py
import logging
import math
# from utils.filter_hub import parents
from nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from config import cfg


logger = logging.getLogger(__name__)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                        -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.01)

def clip_by_norm(layer, norm=1):
    if isinstance(layer, nn.Linear):
        if layer.weight.data.norm(2) > norm:
            layer.weight.data.mul_(norm / layer.weight.data.norm(2).item())

class DecoderLayer(nn.Module):
    def __init__(self, cfg,dim):
        super(DecoderLayer,self).__init__()
        self.joint_ch = dim
        self.self_attn = nn.MultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.scorenet.heads,batch_first = True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.scorenet.heads,batch_first = True)
        self.dropout_rate = 0.1
        feedforward_dim = self.joint_ch * 4
        # MLP
        self.linear1 = nn.Linear(self.joint_ch, feedforward_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear2 = nn.Linear(feedforward_dim, self.joint_ch)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(self.joint_ch)
        self.norm2 = nn.LayerNorm(self.joint_ch)
        self.norm3 = nn.LayerNorm(self.joint_ch)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.ReLU()
        
    def with_pos_embed(self, tensor, pos):
        return tensor + pos
    
    def forward(self, tgt, memory, mask=None, mask_ctx=None, pos=None, pos_ctx=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos[0])
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = tgt2.view(memory.shape[0], -1, self.joint_ch)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos[1]),
                                   key=self.with_pos_embed(memory, pos_ctx),
                                   value=memory)[0]
        tgt2 = tgt2.contiguous().view(tgt.shape[0], tgt.shape[1], tgt.shape[2])
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class ScoreNet(nn.Module):
    def __init__(self, scorehypo_cfg, neighbour_matrix=None):
        super(ScoreNet, self).__init__()
        self.joint_ch = scorehypo_cfg.scorenet.joint_ch
        self.joints = scorehypo_cfg.hyponet.num_joints
        self.num_twists = scorehypo_cfg.hyponet.num_twists
        self.num_item = self.num_twists + self.joints
        self.num_blocks = scorehypo_cfg.scorenet.num_blocks
        self.dropout_rate = 0.25
        self.neighbour_matrix = neighbour_matrix
        self.mask = self.init_mask(self.neighbour_matrix[0])
        self.mask_twist = self.init_mask(self.neighbour_matrix[1])
        self.mask_joints = self.init_mask(self.neighbour_matrix[2])
        self.atten_knn = scorehypo_cfg.scorenet.atten_knn
        self.local_ch = scorehypo_cfg.hrnet.local_ch
        parents = np.array([ 0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11], dtype=np.int64)
        self.parents_idx = torch.tensor(parents[1 : 1+self.num_twists])
        self.child_idx = torch.arange(start=1,end=1+self.num_twists)
        self.emb_h , self.emb_w = cfg.output_hm_shape[1:]
        self.mask_list = []
        assert len(self.atten_knn) == self.num_blocks
        for i in range(self.num_blocks):
            mask_i = np.linalg.matrix_power(neighbour_matrix[3], self.atten_knn[i])
            mask_i = np.array(mask_i!=0, dtype=np.float32)
            mask_i = self.init_mask(mask_i)
            mask_i = 1 - mask_i
            self.mask_list.append(mask_i.bool())
        self.mask_list_cross_atten = [None for i in range(self.num_blocks)]
        
        self.ch = self.joint_ch + self.local_ch
        self.ctx_dim = cfg.feat_dim


        # first layer
        # joints
        self.linear_start_j = nn.Linear(self.joints*3, self.joints * self.joint_ch)
        self.bn_start_j = nn.GroupNorm(32, num_channels=self.joints*self.joint_ch)
        self.activation_start_j = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_start_j = nn.Dropout(p=self.dropout_rate)
        # twist
        self.linear_start_t = nn.Linear(self.num_twists*2, self.num_twists*self.joint_ch)
        self.bn_start_t = nn.GroupNorm(32, num_channels=self.num_twists*self.joint_ch)
        self.activation_start_t = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_start_t = nn.Dropout(p=self.dropout_rate)

        self.ctx_to_x_conv = nn.Conv2d(in_channels=self.ctx_dim, out_channels=self.ch, kernel_size=1)
        self.deconv = make_deconv_layers([self.ctx_dim, 256])

        self.linear_mlp = nn.Sequential(
            nn.Linear(self.num_item*self.ch, 1024),
            nn.GroupNorm(32, num_channels=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=self.dropout_rate),
            # nn.Linear(1024, 1)
            nn.Linear(1024, 256),
            nn.Linear(256, 1)
        )
        #blocks
        self.blocks = nn.ModuleList([DecoderLayer(scorehypo_cfg,self.ch) for i in range(self.num_blocks)])

        self.score_sigmoid = nn.Sigmoid()

    def init_mask(self,neighbour_matrix):
        """
        Only support locally_connected
        """
        #assert self.neighbour_matrix is not None
        L = neighbour_matrix.T
        #assert L.shape == (self.joints, self.joints)
        return torch.from_numpy(L)

    def mask_weights(self, layer, mask, mshape):
        assert isinstance(layer, nn.Linear), 'masked layer must be linear layer'

        output_size, input_size = layer.weight.shape  # pytorch weights [output_channel, input_channel]
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % mshape == 0 and output_size % mshape == 0
        in_F = int(input_size / mshape)
        out_F = int(output_size / mshape)
        weights = layer.weight.data.view([mshape, out_F, mshape, in_F])
        weights.mul_(mask.t().view(mshape, 1, mshape, 1).to(device=weights.get_device()))
    
    def get_local_feature(self, xinj, fmap):
        bs = fmap.shape[0]
        multi_n = xinj.shape[0]//bs
        joint_2d = xinj.view(-1, self.joints,3)[:,:,:2].clone()
        idx_2d = torch.zeros(bs*multi_n,self.num_item,2).to(xinj.device)
        idx_2d[:,:self.joints,0] = joint_2d[:,:,0]
        idx_2d[:,:self.joints,1] = joint_2d[:,:,1]
        idx_2d[:,self.joints:,0] = (joint_2d[:,self.parents_idx,1] + joint_2d[:,self.child_idx,1])*0.5
        idx_2d[:,self.joints:,1] = (joint_2d[:,self.parents_idx,0] + joint_2d[:,self.child_idx,0])*0.5
        idx_2d = idx_2d.view(bs, multi_n, -1, 2)
        local_f = torch.nn.functional.grid_sample(fmap, idx_2d, align_corners=True)
        ctx_local = local_f.transpose(3,1).transpose(2,1)
        return torch.flatten(ctx_local,start_dim=2, end_dim=3)
    
    def forward(self, xinj, ctx=None):
        bs = ctx['global'].shape[0]
        assert xinj.shape[0] % bs == 0
        multi_n = xinj.shape[0] // bs
        whole = bs * multi_n
        xint = xinj[:, 3*self.joints:].clone()
        xinj = xinj[:, :3*self.joints].clone()

        ctx['local'] = self.deconv(ctx['global'])
        
        ctx['local'] = self.get_local_feature(xinj, ctx['local'])

        # mask weights of all linear layers before forward
        self.mask_weights(self.linear_start_j, self.mask_joints, self.joints)

        emb_ctx = positionalencoding2d(self.ch, self.emb_h, self.emb_w).unsqueeze(dim=0).repeat(ctx['global'].size()[0],1,1,1).cuda()
        emb_ctx = torch.flatten(emb_ctx,start_dim=2, end_dim=3).transpose(2,1)
        ctx['global'] = self.ctx_to_x_conv(ctx['global'])
        ctx['global'] = torch.flatten(ctx['global'], start_dim=2, end_dim=3).transpose(2, 1).to(xinj.device)
        emb = get_timestep_embedding(torch.arange(0, self.num_item), self.ch).cuda().view(1, self.num_item, self.ch).to(xinj.device)
        emb_multi = get_timestep_embedding(torch.arange(0, self.num_item), self.ch).cuda().view(1, -1, self.ch)
        emb_multi = emb_multi.unsqueeze(1).repeat(1, multi_n, 1, 1).view(1,-1,self.ch)

        # joints
        x0 = self.linear_start_j(xinj)
        x0 = self.bn_start_j(x0)
        x0 = self.activation_start_j(x0)
        x0 = self.dropout_start_j(x0)
        x0 = x0.contiguous().view(-1, self.joints, self.joint_ch)
        # twist
        x1 = torch.zeros_like(x0)[:, 0:0]
        # combine
        x = torch.cat([x0, x1], dim=1).view(-1, self.num_item, self.joint_ch)
        ctx['local'] = ctx['local'].view(-1, self.num_item, self.joint_ch).to(xinj.device)      # pixel-aligned sample
        x = torch.cat([x, ctx['local']], dim=-1).view(-1, self.num_item, self.ch)

        # construct blocks
        for block_idx in range(self.num_blocks):
            mask_ctx = self.mask_list_cross_atten[block_idx]
            if mask_ctx != None:
                mask_ctx = self.mask_list_cross_atten[block_idx].clone().bool().to(x.device)
            x = self.blocks[block_idx](tgt=x, memory=ctx['global'], pos = [emb, emb_multi], pos_ctx = emb_ctx, mask = self.mask_list[block_idx].clone().bool().to(x.device))

        # final layer
        x = x.view(-1, self.num_item*self.ch)
        score_raw = self.linear_mlp(x).view(bs*multi_n)
        score = self.score_sigmoid(score_raw * 5)
    
        return score_raw, score

def get_scorehypo_score_net(cfg, neighbour_matrix, is_train):
    model = ScoreNet(cfg, neighbour_matrix)
    pretrained = cfg.scorenet.pretrained

    if is_train:
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained lcn model {}'.format(pretrained))
            model.load_state_dict(pretrained_state_dict['state_dict_3d'], strict=True)
        else:
            logger.info('=> init lcn weights from kaiming normal distribution')
            model.apply(init_weights)
            model.apply(clip_by_norm)
    return model
