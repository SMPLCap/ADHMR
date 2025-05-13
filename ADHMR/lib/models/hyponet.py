import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import os, sys
import pickle, h5py
import logging
import math

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

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def clip_by_norm(layer, norm=1):
    if isinstance(layer, nn.Linear):
        if layer.weight.data.norm(2) > norm:
            layer.weight.data.mul_(norm / layer.weight.data.norm(2).item())

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRA, self).__init__()
        self.rank = rank
        self.lora_a = nn.Parameter(torch.randn(in_features, rank))
        self.lora_b = nn.Parameter(torch.randn(rank, out_features))
        self.reset_parameters()
        self.scaling = 1 / (self.rank ** 0.5)
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        return (self.lora_a @ self.lora_b).to(x.device) * self.scaling

class LoRAMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, lora_rank=4, **kwargs):
        super(LoRAMultiheadAttention, self).__init__(embed_dim, num_heads, **kwargs)
        self.lora_proj = LoRA(embed_dim*3, embed_dim, lora_rank)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        # Apply LoRA to in_proj_weight
        adjusted_weight = self.in_proj_weight + self.lora_proj(self.in_proj_weight)

        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "key_padding_mask and attn_mask are not supported with NestedTensor input"
        elif not query.is_nested and key_padding_mask is not None and attn_mask is not None:
            why_not_fast_path = "key_padding_mask and attn_mask were both supplied"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                adjusted_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                adjusted_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights




class LoRAMultiheadAttention_(nn.Module):
    def __init__(self, vit_model, r=4, alpha=16):
        r"""
        vit_model: nn.MultiheadAttention()
        """
        super(LoRAMultiheadAttention, self).__init__()

        assert r > 0

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        w_qkv_linear = vit_model.in_proj_weight
        assert w_qkv_linear.ndim == 2
        self.dim = w_qkv_linear.shape[0] // 3               # input feature dim                    
        self.w_a_linear_q = nn.Parameter(torch.randn(self.dim, r), requires_grad=True)
        self.w_b_linear_q = nn.Parameter(torch.randn(r, self.dim), requires_grad=True)
        self.w_a_linear_v = nn.Parameter(torch.randn(self.dim, r), requires_grad=True)
        self.w_b_linear_v = nn.Parameter(torch.randn(r, self.dim), requires_grad=True)
        
        self.w_As.extend([self.w_a_linear_q, self.w_a_linear_v])
        self.w_Bs.extend([self.w_b_linear_q, self.w_b_linear_v])

        self.reset_parameters()                             # init LoRA params

        vit_model.in_proj_weight[:self.dim, :] = self.apply_lora(vit_model.in_proj_weight[:self.dim, :], self.w_a_linear_q, self.w_b_linear_q)
        vit_model.in_proj_weight[self.dim * 2:, :] = self.apply_lora(vit_model.in_proj_weight[self.dim * 2:, :], self.w_a_linear_v, self.w_b_linear_v)

        self.lora_vit = vit_model
    
    def apply_lora(self, proj_weight, lora_A, lora_B):
        # 通过 LoRA 调整投影矩阵
        lora_proj_weight = proj_weight + self.scaling * (lora_A @ lora_B)
        return lora_proj_weight

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor = None) -> None:
        self.lora_vit.in_proj_weight[:self.dim, :] = self.apply_lora(self.lora_vit.in_proj_weight[:self.dim, :], self.w_a_linear_q, self.w_b_linear_q)
        self.lora_vit.in_proj_weight[self.dim * 2:, :] = self.apply_lora(self.lora_vit.in_proj_weight[self.dim * 2:, :], self.w_a_linear_v, self.w_b_linear_v)

        return self.lora_vit(query, key, value, attn_mask=attn_mask)



class DecoderLayer(nn.Module):
    def __init__(self, cfg, use_lora=False):
        super(DecoderLayer,self).__init__()
        self.local_ch = cfg.hrnet.local_ch
        self.joint_ch = cfg.hyponet.joint_ch + self.local_ch
        self.dropout_rate = 0.1
        self.joints = cfg.hyponet.num_joints
        self.edges = cfg.hyponet.num_twists
        self.num_item = self.edges+self.joints
        self.use_lora = use_lora
        feedforward_dim = self.joint_ch * 4

        self.norm1 = nn.LayerNorm(self.joint_ch)
        if self.use_lora:
            self.self_attn = LoRAMultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.hyponet.heads, lora_rank=4, batch_first=True)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.hyponet.heads, batch_first = True)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        self.norm2 = nn.LayerNorm(self.joint_ch)
        if self.use_lora:
            self.multihead_attn = LoRAMultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.hyponet.heads, lora_rank=4, batch_first = True)
        else:
            self.multihead_attn = nn.MultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.hyponet.heads,batch_first = True)
        self.dropout2 = nn.Dropout(p=self.dropout_rate) 

        self.linear1 = nn.Linear(self.joint_ch, feedforward_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear2 = nn.Linear(feedforward_dim, self.joint_ch)
        self.norm3 = nn.LayerNorm(self.joint_ch)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor + pos
    
    def forward(self, tgt, memory, mask= None, mask_ctx = None, pos= None, pos_ctx=None, gen_multi=False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        if gen_multi:
            bs = memory.shape[0]
            multi_n = tgt.shape[0] // bs
            pos = pos.repeat(1,multi_n,1).view(-1,self.num_item*multi_n,self.joint_ch)
            tgt2 = self.norm2(tgt).view(bs,-1,self.joint_ch)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos),
                                    key=self.with_pos_embed(memory, pos_ctx),
                                    value=memory, attn_mask=None)[0].contiguous().view(bs*multi_n,-1,self.joint_ch)
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos),
                                    key=self.with_pos_embed(memory, pos_ctx),
                                    value=memory, attn_mask=None)[0]
            tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class HypoNet(nn.Module):
    def __init__(self, cfg, neighbour_matrix=None, use_lora=False):
        super(HypoNet, self).__init__()
        self.joint_ch = cfg.hyponet.joint_ch
        self.local_ch = cfg.hrnet.local_ch
        self.ch = cfg.hyponet.joint_ch + self.local_ch
        self.joints = cfg.hyponet.num_joints
        self.edges = cfg.hyponet.num_twists
        self.num_item = self.edges+self.joints
        self.num_blocks = cfg.hyponet.num_blocks
        self.dropout_rate = 0.25
        # mask_matrix
        self.neighbour_matrix = neighbour_matrix
        self.mask = self.init_mask(self.neighbour_matrix[0])
        self.mask_twist = self.init_mask(self.neighbour_matrix[1])
        self.mask_joints = self.init_mask(self.neighbour_matrix[2])
        self.mask_list = []
        self.atten_knn = cfg.hyponet.atten_knn
        assert len(self.atten_knn) == self.num_blocks
        for i in range(self.num_blocks):
            mask_i = np.linalg.matrix_power(neighbour_matrix[3], self.atten_knn[i])
            mask_i = np.array(mask_i!=0, dtype=np.float32)
            mask_i = self.init_mask(mask_i)
            mask_i = 1 - mask_i
            self.mask_list.append(mask_i.bool())

        # first layer
        self.linear_start_j = nn.Linear(self.joints*3, self.joints*self.joint_ch)
        self.bn_start_j = nn.GroupNorm(32, num_channels=self.joints*self.joint_ch)
        self.activation_start_j = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_start_j = nn.Dropout(p=self.dropout_rate)
        # twist
        self.linear_start_t = nn.Linear(self.edges*2, self.edges*self.joint_ch)
        self.bn_start_t = nn.GroupNorm(32, num_channels=self.edges*self.joint_ch)
        self.activation_start_t = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_start_t = nn.Dropout(p=self.dropout_rate)

        self.emb_h = self.emb_w = 8

        # final layer
        self.linear_final_j = nn.Linear(self.joints*self.ch, self.joints*3)
        self.linear_final_t = nn.Linear(self.edges*self.ch, self.edges*2)
         
        #blocks
        self.blocks = nn.ModuleList([DecoderLayer(cfg, use_lora) for i in range(self.num_blocks)])

        # time
        self.temb_ch =cfg.hyponet.temb_ch 
        self.temb_dense = nn.ModuleList([torch.nn.Linear(self.temb_ch,self.temb_ch*4),
                                        torch.nn.Linear(self.temb_ch*4,self.ch),])

    def init_mask(self,neighbour_matrix):
        """
        Only support locally_connected
        """
        L = neighbour_matrix.T
        return torch.from_numpy(L)

    def mask_weights(self, layer,mask,mshape):
        assert isinstance(layer, nn.Linear), 'masked layer must be linear layer'

        output_size, input_size = layer.weight.shape  # pytorch weights [output_channel, input_channel]
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % mshape == 0 and output_size % mshape == 0
        in_F = int(input_size / mshape)
        out_F = int(output_size / mshape)
        weights = layer.weight.data.view([mshape, out_F, mshape, in_F])
        weights.mul_(mask.t().view(mshape, 1, mshape, 1).to(device=weights.get_device()))
    
    def forward(self, xinj, xint, t, ctx=None, gen_multi=False):
        bs = ctx['global'].shape[0]
        if gen_multi:
            assert xinj.shape[0]%bs==0
            multi_n = xinj.shape[0] // bs
        self.mask_weights(self.linear_start_j, self.mask_joints, self.joints)         # 利用edge和joint的连接关系mask掉mlp中不需要的neuron
        self.mask_weights(self.linear_start_t, self.mask_twist, self.edges)
        self.mask_weights(self.linear_final_j, self.mask_joints, self.joints)
        self.mask_weights(self.linear_final_t, self.mask_twist, self.edges)

        # condition 
        emb_ctx = positionalencoding2d(self.ch, self.emb_h, self.emb_w).unsqueeze(dim=0).cuda()
        emb_ctx = torch.flatten(emb_ctx, start_dim=2, end_dim=3).transpose(2,1)             # 1, 64, 512
        ctx_global = torch.flatten(ctx['global'], start_dim=2, end_dim=3).transpose(2,1)    # bs, 64, 512
        ctx_local = ctx['local'].view(-1, self.num_item, self.local_ch)                     # bs, 52, 256
        if gen_multi:
            ctx_local = ctx_local.unsqueeze(1).repeat(1,multi_n,1,1).view(-1,self.num_item,self.local_ch)
        emb = get_timestep_embedding(torch.arange(0, self.num_item), self.ch).cuda().view(1,self.num_item,self.ch).cuda()   # 1, 52, 512

        # time
        temb = get_timestep_embedding(t, self.temb_ch)
        temb = self.temb_dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense[1](temb)
        temb = nonlinearity(temb)
        temb = temb.unsqueeze(1).repeat(1,self.num_item,1)
        temb = temb.view(-1,self.num_item,self.ch)                         # bs, 52, 512

        # first layer
        # joints
        x0 = self.linear_start_j(xinj)
        x0 = self.bn_start_j(x0)
        x0 = self.activation_start_j(x0)
        x0 = self.dropout_start_j(x0)
        x0 = x0.contiguous().view(-1, self.joints, self.joint_ch)           # bs, 29, 256
        # twist
        x1 = self.linear_start_t(xint)
        x1 = self.bn_start_t(x1)
        x1 = self.activation_start_t(x1)
        x1 = self.dropout_start_t(x1)
        x1 =  x1.contiguous().view(-1, self.edges,self.joint_ch)           # bs, 23, 256

        x = torch.cat([x0, x1], dim=1).view(-1, self.num_item, self.joint_ch)
        x = torch.cat([x, ctx_local], dim=-1).view(-1, self.num_item, self.ch)               # bs, 52, 512
        x += temb

        for block_idx in range(self.num_blocks):
            x = self.blocks[block_idx](tgt=x, memory=ctx_global, pos = emb, pos_ctx = emb_ctx, \
                                       mask = self.mask_list[block_idx].clone().bool().to(x.device), \
                                        gen_multi=gen_multi)

        x = x.view(-1,self.num_item,self.ch)
        xj = x[:,:self.joints,:].view(-1, self.joints*self.ch)
        xt = x[:,self.joints:,:].view(-1, self.edges*self.ch)
        xj = self.linear_final_j(xj) 
        xj = xj + xinj
        xt = self.linear_final_t(xt) 
        xt = xt + xint
        return xj, xt

def get_hyponet(cfg, neighbour_matrix, is_train, **kwargs):
    model = HypoNet(cfg, neighbour_matrix, **kwargs)
    pretrained = cfg.hyponet.pretrained

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
