import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, x.shape[-1]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, x, x_fake, given_y=None, given_i=None, given_w=None, \
                             nd_to_sample=None, data_parallel=None, \
                             ed_to_sample=None):
    indices = nd_to_sample, ed_to_sample
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    u = torch.FloatTensor(x.shape[0], 1, 1).to(device)
    u.data.resize_(x.shape[0], 1, 1)
    u.uniform_(0, 1)
    x_both = x.data*u + x_fake.data*(1-u)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    if data_parallel:
        _output = data_parallel(D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        _output = D(x_both, given_i, given_y, given_w, nd_to_sample)
    grad = torch.autograd.grad(outputs=_output, inputs=x_both, grad_outputs=grad_outputs, \
                               retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, batch_norm=False):
    block = []
    
    if upsample:
        if spec_norm:
            block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True)))
        else:
            block.append(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True)))
        else:        
            block.append(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True))
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(inplace=True))
    # elif "tanh":
    #     block.append(torch.nn.Tanh())
    return block


class EncoderDecoder(nn.Module):
    def __init__(self, n_updown, in_channels):
        super(EncoderDecoder, self).__init__()
        layers = []
        for _ in range(n_updown):
            layers += conv_block(3*in_channels, 3*in_channels, 3, 2, 1, act="relu", batch_norm=True)
        for _ in range(n_updown):
            layers += conv_block(3*in_channels, 3*in_channels, 4, 2, 1, act="relu", batch_norm=True, upsample=True)
        layers += conv_block(3*in_channels, in_channels, 3, 1, 1, act="relu", batch_norm=True)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_updown, in_channels):
        super(UNet, self).__init__()
        self.inc = DoubleConv(3*in_channels, 3*in_channels)
        self.down_path = nn.ModuleList()
        for _ in range(n_updown):
            self.down_path.append(Down(3*in_channels, 3*in_channels))

        self.up_path = nn.ModuleList()    
        for _ in range(n_updown):
            self.up_path.append(Up(6*in_channels, 3*in_channels))
        self.last_layer = OutConv(3*in_channels, in_channels)

    def forward(self, x):
        blocks = []
        blocks.append(self.inc(x))
        for i, down in enumerate(self.down_path):
            x_d = down(blocks[i])
            blocks.append(x_d)
        y = blocks[-1]
        for up, x_d in zip(self.up_path, blocks[:-1][::-1]):
            y = up(y, x_d)
        return self.last_layer(y)

# ORIGINAL
class CMP(nn.Module):
    def __init__(self, n_updown, in_channels):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky"))
        # self.ED = UNet(n_updown, in_channels)
             
    def forward(self, feats, edges=None):
        
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        

        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = torch.scatter_add(pooled_v_pos, 0, pos_v_dst, pos_vecs_src)

        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = torch.scatter_add(pooled_v_neg, 0, neg_v_dst, neg_vecs_src)

        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        # out = self.ED(enc_in)
        return out


# ## FOR ONNX
# def _scatter_add_loop(pooled_v, inds, v_src):
#     pooled_v = torch.zeros_like(pooled_v)
#     for _to in range(len(pooled_v)):
#         _from_v = v_src[np.where(_to == inds)]
#         pooled_v[_to] += _from_v.sum(0)
#     return pooled_v

# class CMP(nn.Module):
#     def __init__(self, n_updown, in_channels):
#         super(CMP, self).__init__()
#         self.in_channels = in_channels
#         self.encoder = nn.Sequential(
#             *conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
#             *conv_block(2*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
#             *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky"))
#         # self.ED = UNet(n_updown, in_channels)
             
#     def forward(self, feats, edges=None):
        
#         # allocate memory
#         dtype, device = feats.dtype, feats.device
#         edges = edges.view(-1, 3)
#         V, E = feats.size(0), edges.size(0)
#         pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
#         pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        

#         # pool positive edges
#         pos_inds = torch.where(edges[:, 1] > 0)
#         pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
#         pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
#         pos_vecs_src = feats[pos_v_src.contiguous()]
#         pooled_v_pos = _scatter_add_loop(pooled_v_pos, pos_v_dst, pos_vecs_src)

#         # pool negative edges
#         neg_inds = torch.where(edges[:, 1] < 0)
#         neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
#         neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
#         neg_vecs_src = feats[neg_v_src.contiguous()]
#         pooled_v_neg = _scatter_add_loop(pooled_v_neg, neg_v_dst, neg_vecs_src)

#         # update nodes features
#         enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
#         out = self.encoder(enc_in)
#         # out = self.ED(enc_in)
#         return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        enc_blocks = []
        enc_blocks += conv_block(1, 32, 3, 2, 1, act="relu", batch_norm=True)
        enc_blocks += conv_block(32, 64, 3, 2, 1, act="relu", batch_norm=True)
        enc_blocks += conv_block(64, 128, 3, 2, 1, act="relu", batch_norm=True)
        enc_blocks += conv_block(128, 256, 3, 2, 1, act="relu", batch_norm=True)

        dec_blocks = []
        dec_blocks += conv_block(256, 128, 4, 2, 1, act="relu", upsample=True, batch_norm=True)
        dec_blocks += conv_block(128, 64, 4, 2, 1, act="relu", upsample=True, batch_norm=True)
        dec_blocks += conv_block(64, 32, 4, 2, 1, act="relu", upsample=True, batch_norm=True)
        dec_blocks += conv_block(32, 32, 4, 2, 1, act="relu", upsample=True, batch_norm=True)
        dec_blocks += conv_block(32, 1, 3, 1, 1, act="none")

        self.enc = nn.Sequential(*enc_blocks)
        self.dec = nn.Sequential(*dec_blocks)

    def forward(self, x):
        x = self.enc(x)
        y = self.dec(x)
        return y, x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(146, 16 * self.init_size ** 2)) #146
        self.upsample_1 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_2 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_3 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        # self.upsample_4 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        # self.upsample_5 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.cmp_1 = CMP(n_updown= 3, in_channels=16)
        self.cmp_2 = CMP(n_updown= 4, in_channels=16)
        self.cmp_3 = CMP(n_updown= 5, in_channels=16)
        self.cmp_4 = CMP(n_updown= 6, in_channels=16)
        self.cmp_5 = CMP(n_updown= 7, in_channels=16)
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 1, 1, act="leaky"),
            *conv_block(256, 128, 3, 1, 1, act="leaky"),    
            *conv_block(128, 1, 3, 1, 1, act="tanh"))                                        
        

        # for finetuning
        self.l1_fixed = nn.Sequential(nn.Linear(1, 1 * self.init_size ** 2))
        self.enc_1 = nn.Sequential(
            *conv_block(2, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 16, 3, 2, 1, act="leaky"))
            # *conv_block(32, 16, 3, 2, 1, act="leaky"))   
        self.enc_2 = nn.Sequential(
            *conv_block(32, 32, 3, 1, 1, act="leaky"),
            *conv_block(32, 16, 3, 1, 1, act="leaky"))   

    def forward(self, z, given_m=None, given_y=None, given_w=None, given_v=None):
        z = z.view(-1, 128)
        # include nodes
        if True:
            print(given_y.shape)
            y = given_y.view(-1, 18) #10
            z = torch.cat([z, y], 1)
        x = self.l1(z)      
        f = x.view(-1, 16, self.init_size, self.init_size)

        # combine masks and noise vectors
        m = self.enc_1(given_m)
        f = torch.cat([f, m], 1)
        f = self.enc_2(f)

        # apply Conv-MPN
        x = self.cmp_1(f, given_w).view(-1, *f.shape[1:])
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])   
        x = self.upsample_2(x)
        x = self.cmp_3(x, given_w).view(-1, *x.shape[1:])   
        x = self.upsample_3(x)
        # x = self.cmp_4(x, given_w).view(-1, *x.shape[1:])   
        # x = self.upsample_4(x)
        # x = self.cmp_5(x, given_w).view(-1, *x.shape[1:])   
        # x = self.upsample_5(x)
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, *x.shape[2:])
        
        return x

# BACKUP
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.init_size = 32 // 4
#         self.l1 = nn.Sequential(nn.Linear(138, 16 * self.init_size ** 2))
#         self.upsample_1 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
#         self.upsample_2 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
#         self.cmp_1 = CMP(in_channels=16)
#         self.cmp_2 = CMP(in_channels=16)
#         self.decoder = nn.Sequential(
#             *conv_block(16, 256, 3, 1, 1, act="leaky"),
#             *conv_block(256, 128, 3, 1, 1, act="leaky"),    
#             *conv_block(128, 1, 3, 1, 1, act="tanh"))                                        
        

#         # for finetuning
#         self.l1_fixed = nn.Sequential(nn.Linear(1, 1 * self.init_size ** 2))
#         self.enc_1 = nn.Sequential(
#             *conv_block(1, 32, 3, 2, 1, act="leaky"),
#             *conv_block(32, 16, 3, 2, 1, act="leaky"))   
#         self.enc_2 = nn.Sequential(
#             *conv_block(32, 32, 3, 1, 1, act="leaky"),
#             *conv_block(32, 16, 3, 1, 1, act="leaky"))   

#     def forward(self, z, given_i=None, given_m=None, given_y=None, given_w=None, given_v=None, state=None):
#         z = z.view(-1, 128)
#         # include nodes
#         if True:
#             y = given_y.view(-1, 10)
#             z = torch.cat([z, y], 1)
#         x = self.l1(z)      
#         f = x.view(-1, 16, self.init_size, self.init_size)

#         # combine masks and noise vectors
#         m = self.enc_1(given_m.unsqueeze(1))
#         f = torch.cat([f, m], 1)
#         f = self.enc_2(f)

#         # apply Conv-MPN
#         x = self.cmp_1(f, given_w).view(-1, *f.shape[1:])
#         x = self.upsample_1(x)
#         x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])   
#         x = self.upsample_2(x)
#         x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
#         x = x.view(-1, *x.shape[2:])
#         return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            *conv_block(9, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))

        self.l1 = nn.Sequential(nn.Linear(18, 8 * 64 ** 2))
        self.cmp_1 = CMP(n_updown= 5, in_channels=16)
        self.downsample_1 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_2 = CMP(n_updown= 4, in_channels=16)
        self.downsample_2 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_3 = CMP(n_updown= 3, in_channels=16)
        self.downsample_3 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        # self.cmp_4 = CMP(in_channels=16)
        # self.downsample_4 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        # self.cmp_5 = CMP(in_channels=16)
        # self.downsample_5 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))

        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 2, 1, act="leaky"),
            *conv_block(256, 128, 3, 2, 1, act="leaky"),
            *conv_block(128, 128, 3, 2, 1, act="leaky"))
        
        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))
        self.fc_layer_local = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, given_i=None, given_y=None, given_w=None, nd_to_sample=None):
        x = x.view(-1, 1, 64, 64)
        # include nodes
        if True:
            y = given_y
            y = self.l1(y)
            y = y.view(-1, 8, 64, 64)
            x = torch.cat([x, y], 1)

        x = self.encoder(x)
        x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])  
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_2(x)
        x = self.cmp_3(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_3(x)
        # x = self.cmp_4(x, given_w).view(-1, *x.shape[1:])
        # x = self.downsample_4(x)
        # x = self.cmp_5(x, given_w).view(-1, *x.shape[1:])
        # x = self.downsample_5(x)


        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, x.shape[1])
        
        # global loss
        x_g = add_pool(x, nd_to_sample)
        validity_global = self.fc_layer_global(x_g)

        # local loss
        if True:
            x_loc = self.fc_layer_local(x)
            validity_local = add_pool(x_loc, nd_to_sample)
            # print('(gl: {}) (lc: {})'.format(validity_global, validity_local))
            validity = validity_global+5*validity_local
            return validity
        else:
            return validity_global
    
