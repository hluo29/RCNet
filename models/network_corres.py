from turtle import forward
import numpy as np
import functools
from requests import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# from backbone.resnet import ResNet50
# try:
#     from models.modules.dcn_v2 import DCN_sep, DCN
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')

from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation
from models.module_util import ResidualBlock_noBN_noAct, make_layer


class Conv_relu(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size, stride, padding, has_relu=True, efficient=False):
        super(Conv_relu, self).__init__()
        self.has_relu = has_relu
        self.efficient = efficient

        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        def _func_factory(conv, relu, has_relu):
            def func(x):
                x = conv(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(self.conv, self.relu, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x

class Spatial(nn.Module):
    def __init__(self, nf, n_group=8, from_align=False):
        super(Spatial, self).__init__()
        self.from_align = from_align
        self.g = n_group
        self.k = 3        

        self.conv1 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.conv2 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_cha = nn.Sequential(
            nn.Linear(nf, nf//2),
            nn.ReLU(inplace=True),
            nn.Linear(nf//2, nf),
            nn.Sigmoid()
        )
        if not self.from_align:
            self.conv_spa = nn.Sequential(
                Conv_relu(nf, nf, (5, 1), 1, (2, 0), has_relu=True),
                Conv_relu(nf, nf, (1, 5), 1, (0, 2), has_relu=True)
            )
        else:
            self.conv_spa = nn.Sequential(
                Conv_relu(nf, nf, (5, 1), 1, (2, 0), has_relu=True),
                Conv_relu(nf, nf, (1, 5), 1, (0, 2), has_relu=True)
            )
            self.nn_conv = Conv_relu(nf, self.g * self.k ** 2, self.k, 1, (self.k-1)//2, has_relu=False)
            
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        B, N, C, H, W = x.size()
        x = x.view(-1, C, H, W)

        x = self.conv2(self.conv1(x))
        fea_cha = self.avg_pool(x).view(-1, C)
        fea_cha = self.conv_cha(fea_cha).view(-1, C, 1, 1)
        if not self.from_align:
            fea_spa = self.conv_spa(x)
        else:
            fea_l = []
            x = x.view(B, N, C, H, W)
            for i in range(N):
                fea_spa = self.conv_spa(x[:, i, :, :, :])
                fea_spa = self.nn_conv(fea_spa).view(B, self.g, 1, self.k ** 2, H, W)
                fea_spa = self.relu((fea_spa * mask[:, i, :, :, :]).sum(dim=3).view(B, C, H, W))
                fea_l.append(fea_spa)
            fea_spa = torch.stack(fea_l, dim=1).view(-1, C, H, W)
            x = x.view(-1, C, H, W)
        fea_fuse = fea_cha * fea_spa + x
        fea_fuse = fea_fuse.view(B, -1, C, H, W)
        
        return fea_fuse

class Recura(nn.Module):
    def __init__(self, nf, n_view=3, from_align=False):
        super(Recura, self).__init__()
        self.n_view = n_view
        self.from_align = from_align

        self.spa = Spatial(nf=nf, from_align=from_align)
        self.trans_conv = nn.Conv2d(nf, 3, 1, 1, bias=True)

    def forward(self, x, mask=None):
        # L1_fea, L2_fea, L3_fea = x
        # center = self.n_view // 2

        spa_fea = []
        if not self.from_align:
            for i in range(self.n_view):
                spa_fea.append(self.spa(x[i]))
        else:
            for i in range(self.n_view):
                spa_fea.append(self.spa(x[i], mask[i]))

        recur_img = self.trans_conv(spa_fea[0][:, 1, :, :, :])
        
        return spa_fea, recur_img

class Aggregate(nn.Module):
    def __init__(self, nf=64, nbr=4, n_group=8, kernels=[3, 3, 3, 3], patches=[7, 11, 15], cor_ksize=3, adding_avg=True, to_enhance=True):
        super(Aggregate, self).__init__()
        self.nbr = nbr
        self.cas_k = kernels[0]
        self.k1 = kernels[1]
        self.k2 = kernels[2]
        self.k3 = kernels[3]
        self.g = n_group
        self.adding_avg = adding_avg
        self.to_enhance = to_enhance

        self.L3_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L3_conv2 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L3_conv3 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L3_conv4 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L3_mask = Conv_relu(nf, self.g * self.k3 ** 2, self.k3, 1, (self.k3-1)//2, has_relu=False)
        self.L3_avg1 = Conv_relu(3, nf, 3, 1, 1, has_relu=True)
        self.L3_avg2 = Conv_relu(nf, self.k3 ** 2, 1, 1, 0)
        self.L3_nn_conv = Conv_relu(nf * (self.nbr+1), nf, 3, 1, 1, has_relu=True)

        self.L2_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L2_conv2 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L2_conv3 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L2_conv4 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L2_conv5 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L2_mask = Conv_relu(nf, self.g * self.k2 ** 2, self.k2, 1, (self.k2-1)//2, has_relu=False)
        self.L2_avg1 = Conv_relu(3, nf, 3, 1, 1, has_relu=True)
        self.L2_avg2 = Conv_relu(nf, self.k3 ** 2, 1, 1, 0)
        self.L2_nn_conv = Conv_relu(nf * (self.nbr+1), nf, 3, 1, 1, has_relu=True)
        self.L2_fea_conv = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

        self.L1_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L1_conv2 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L1_conv3 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L1_conv4 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L1_conv5 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L1_mask = Conv_relu(nf, self.g * self.k1 ** 2, self.k1, 1, (self.k1-1)//2, has_relu=False)
        self.L1_avg1 = Conv_relu(3, nf, 3, 1, 1, has_relu=True)
        self.L1_avg2 = Conv_relu(nf, self.k3 ** 2, 1, 1, 0)
        self.L1_nn_conv = Conv_relu(nf * (self.nbr+1), nf, 3, 1, 1, has_relu=True)
        self.L1_fea_conv = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.patch_size = patches
        self.cor_k = cor_ksize
        self.padding = (self.cor_k - 1) // 2
        self.pad_size = [self.padding + (patch - 1) // 2 for patch in self.patch_size]
        self.add_num = [2 * pad - self.cor_k + 1 for pad in self.pad_size]
        self.L3_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[0],
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)
        self.L2_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[1],
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)
        self.L1_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch_size[2],
                stride=1, padding=self.padding, dilation=1, dilation_patch=1)

    def forward(self, nbr_fea_l, ref_fea_l, recur_fea_l):
        # L3
        B, C, H, W = nbr_fea_l[2].size()
        L3_w = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_w = self.L3_conv4(self.L3_conv3(self.L3_conv2(self.L3_conv1(L3_w))))
        L3_mask = self.L3_mask(L3_w).view(B, self.g, 1, self.k3 ** 2, H, W)   
        # corr: B, (2 * dis + 1) ** 2, H, W
        L3_norm_ref_fea = F.normalize(ref_fea_l[2], dim=1)
        L3_norm_nbr_fea = F.normalize(nbr_fea_l[2], dim=1)
        L3_corr = self.L3_corr(L3_norm_ref_fea, L3_norm_nbr_fea).view(B, -1, H, W)
        # corr_ind: B, H, W
        _, L3_corr_ind = torch.topk(L3_corr, self.nbr, dim=1)
        L3_corr_ind = L3_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L3_ind_row_add = L3_corr_ind // self.patch_size[0] * (W + self.add_num[0])
        L3_ind_col_add = L3_corr_ind % self.patch_size[0]
        L3_corr_ind = L3_ind_row_add + L3_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        L3_lt_ind = y * (W + self.add_num[0]) + x
        L3_lt_ind = L3_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L3_corr_ind = (L3_corr_ind + L3_lt_ind).view(-1)
        # L3_nbr: B, 64 * k * k, (H + 2 * pad - k + 1) * (W + 2 * pad -k + 1)
        L3_nbr = F.unfold(nbr_fea_l[2], self.cor_k, dilation=1, padding=self.pad_size[0], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        # L3: B * H * W * nbr, 64 * k * k
        L3 = L3_nbr[ind_B, :, L3_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        if self.to_enhance:
            L3_corr_mask = L3[:, 0:C, :, :]
            L3_corr_mask = L3_corr_mask.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
            L3_corr_mask = L3_corr_mask.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        if self.adding_avg:
            L3_avg = (L3[:, 0:C, :, :] + L3[:, C:2*C, :, :] + L3[:, 2*C:3*C, :, :] + L3[:, 3*C:4*C, :, :])/4.0
            L3_avg_mask = self.L3_avg2(self.L3_avg1(recur_fea_l[2]))
            L3_avg_mask = L3_avg_mask.permute(0, 2, 3, 1).reshape(B * H * W, 1, self.cor_k, self.cor_k)
            L3_avg = L3_avg * L3_avg_mask
            L3 = torch.cat([L3, L3_avg], dim=1)       
        L3 = self.L3_nn_conv(L3)
        L3 = L3.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L3 = L3.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L3 = self.relu((L3 * L3_mask).sum(dim=3).view(B, C, H, W))
        L3_fuse_fea = L3

        # L2
        B, C, H, W = nbr_fea_l[1].size()
        L2_w = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_w = self.L2_conv1(L2_w)
        L3_w = F.interpolate(L3_w, scale_factor=2, mode='bilinear', align_corners=False)
        L2_w = self.L2_conv5(self.L2_conv4(self.L2_conv3(self.L2_conv2(torch.cat([L2_w, L3_w], dim=1)))))
        L2_mask = self.L2_mask(L2_w).view(B, self.g, 1, self.k2 ** 2, H, W)
        # generate most similar feas
        L2_norm_ref_fea = F.normalize(ref_fea_l[1], dim=1)
        L2_norm_nbr_fea = F.normalize(nbr_fea_l[1], dim=1)
        L2_corr = self.L2_corr(L2_norm_ref_fea, L2_norm_nbr_fea).view(B, -1, H, W)
        _, L2_corr_ind = torch.topk(L2_corr, self.nbr, dim=1)
        L2_corr_ind = L2_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L2_ind_row_add = L2_corr_ind // self.patch_size[1] * (W + self.add_num[1])
        L2_ind_col_add = L2_corr_ind % self.patch_size[1]
        L2_corr_ind = L2_ind_row_add + L2_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        L2_lt_ind = y * (W + self.add_num[1]) + x
        L2_lt_ind = L2_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L2_corr_ind = (L2_corr_ind + L2_lt_ind).view(-1)
        L2_nbr = F.unfold(nbr_fea_l[1], self.cor_k, dilation=1, padding=self.pad_size[1], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        L2 = L2_nbr[ind_B, :, L2_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        if self.to_enhance:
            L2_corr_mask = L2[:, 0:C, :, :]
            L2_corr_mask = L2_corr_mask.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
            L2_corr_mask = L2_corr_mask.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        if self.adding_avg:
            L2_avg = (L2[:, 0:C, :, :] + L2[:, C:2*C, :, :] + L2[:, 2*C:3*C, :, :] + L2[:, 3*C:4*C, :, :])/4.0
            L2_avg_mask = self.L2_avg2(self.L2_avg1(recur_fea_l[1]))
            L2_avg_mask = L2_avg_mask.permute(0, 2, 3, 1).reshape(B * H * W, 1, self.cor_k, self.cor_k)
            L2_avg = L2_avg * L2_avg_mask
            L2 = torch.cat([L2, L2_avg], dim=1)
        L2 = self.L2_nn_conv(L2)
        L2 = L2.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L2 = L2.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L2 = self.relu((L2 * L2_mask).sum(dim=3).view(B, C, H, W))
        # fuse F2 with F3
        L3 = F.interpolate(L3, scale_factor=2, mode='bilinear', align_corners=False)
        L2 = self.L2_fea_conv(torch.cat([L2, L3], dim=1))
        L2_fuse_fea = L2

        # L1
        B, C, H, W = nbr_fea_l[0].size()
        L1_w = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_w = self.L1_conv1(L1_w)
        L2_w = F.interpolate(L2_w, scale_factor=2, mode='bilinear', align_corners=False)
        L1_w = self.L1_conv5(self.L1_conv4(self.L1_conv3(self.L1_conv2(torch.cat([L1_w, L2_w], dim=1)))))
        L1_mask = self.L1_mask(L1_w).view(B, self.g, 1, self.k1 ** 2, H, W)
        # generate mot similar feas
        L1_norm_ref_fea = F.normalize(ref_fea_l[0], dim=1)
        L1_norm_nbr_fea = F.normalize(nbr_fea_l[0], dim=1)
        L1_corr = self.L1_corr(L1_norm_ref_fea, L1_norm_nbr_fea).view(B, -1, H, W)
        _, L1_corr_ind = torch.topk(L1_corr, self.nbr, dim=1)
        L1_corr_ind = L1_corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)
        L1_ind_row_add = L1_corr_ind // self.patch_size[2] * (W + self.add_num[2])
        L1_ind_col_add = L1_corr_ind % self.patch_size[2]
        L1_corr_ind = L1_ind_row_add + L1_ind_col_add
        # generate top-left indexes
        y = torch.arange(H).repeat_interleave(W).cuda()
        x = torch.arange(W).repeat(H).cuda()
        L1_lt_ind = y * (W + self.add_num[2]) + x
        L1_lt_ind = L1_lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        L1_corr_ind = (L1_corr_ind + L1_lt_ind).view(-1)
        L1_nbr = F.unfold(nbr_fea_l[0], self.cor_k, dilation=1, padding=self.pad_size[2], stride=1)
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W * self.nbr).cuda()
        L1 = L1_nbr[ind_B, :, L1_corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        if self.to_enhance:
            L1_corr_mask = L1[:, 0:C, :, :]
            L1_corr_mask = L1_corr_mask.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
            L1_corr_mask = L1_corr_mask.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        # L1 = L1.permute(0, 2, 1, 3, 4).contiguous().view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        if self.adding_avg:
            L1_avg = (L1[:, 0:C, :, :] + L1[:, C:2*C, :, :] + L1[:, 2*C:3*C, :, :] + L1[:, 3*C:4*C, :, :])/4.0
            L1_avg_mask = self.L1_avg2(self.L1_avg1(recur_fea_l[0]))
            L1_avg_mask = L1_avg_mask.permute(0, 2, 3, 1).reshape(B * H * W, 1, self.cor_k, self.cor_k)
            L1_avg = L1_avg * L1_avg_mask
            L1 = torch.cat([L1, L1_avg], dim=1)
        L1 = self.L1_nn_conv(L1)
        L1 = L1.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
        L1 = L1.view(B, self.g, C // self.g, self.cor_k ** 2, H, W)
        L1 = self.relu((L1 * L1_mask).sum(dim=3).view(B, C, H, W))
        # fuse L1 with L2
        L2 = F.interpolate(L2, scale_factor=2, mode='bilinear', align_corners=False)
        L1 = self.L1_fea_conv(torch.cat([L1, L2], dim=1))
        L1_fuse_fea = L1

        return [L1_fuse_fea, L2_fuse_fea, L3_fuse_fea, L1_corr_mask, L2_corr_mask, L3_corr_mask]

class Fusion(nn.Module):
    def __init__(self, nf, n_view=3):
        super(Fusion, self).__init__()
        self.n_view = n_view

        self.ref_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.nbr_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.fuse_conv = nn.Conv2d(nf * n_view, nf * n_view, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()

        emb_ref = self.ref_conv(x[:, N // 2, :, :, :].clone())
        emb = self.nbr_conv(x.view(-1, C, H, W)).view(B, N, C, H, W)

        cor_l = []
        for i in range(N):
            cor = torch.sum(emb[:, i, :, :, :] * emb_ref, dim=1, keepdim=True)
            cor_l.append(cor)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aggr_fea = x.view(B, -1, H, W) * cor_prob

        out = self.lrelu(self.fuse_conv(aggr_fea)).view(B, N, -1, H, W)

        return out

class Recurb(nn.Module):
    def __init__(self, nf, n_view, nbr, n_group, kernels, patches, cor_ksize):
        super(Recurb, self).__init__()
        self.n_view = n_view
        
        self.aggr = Aggregate(nf=nf, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches, cor_ksize=cor_ksize)
        self.fuse = Fusion(nf=nf)
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.trans_conv = nn.Conv2d(nf, 3, 1, 1, bias=True)

    def forward(self, x, recur_x):
        L1_fea, L2_fea, L3_fea = x
        recur_fea_l = [
            recur_x, self.avg1(recur_x), self.avg2(self.avg1(recur_x))
        ]
        center = self.n_view // 2

        ref_fea_l = [
            L1_fea[:, center, :, :, :].clone(), L2_fea[:, center, :, :, :].clone(),
            L3_fea[:, center, :, :, :].clone()
        ]
        # 
        aggr_fea_l1, aggr_fea_l2, aggr_fea_l3 = [], [], []
        aggr_mask_l1, aggr_mask_l2, aggr_mask_l3 = [], [], []
        for i in range(self.n_view):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aggr_l1, aggr_l2, aggr_l3, mask_l1, mask_l2, mask_l3 = self.aggr(nbr_fea_l, ref_fea_l, recur_fea_l)
            aggr_fea_l1.append(aggr_l1)
            aggr_fea_l2.append(aggr_l2)
            aggr_fea_l3.append(aggr_l3)
            # 
            aggr_mask_l1.append(mask_l1)
            aggr_mask_l2.append(mask_l2)
            aggr_mask_l3.append(mask_l3)
        
        aggr_fea_l1 = torch.stack(aggr_fea_l1, dim=1)
        aggr_fea_l2 = torch.stack(aggr_fea_l2, dim=1)
        aggr_fea_l3 = torch.stack(aggr_fea_l3, dim=1)
        #
        aggr_mask_l1 = torch.stack(aggr_mask_l1, dim=1)
        aggr_mask_l2 = torch.stack(aggr_mask_l2, dim=1)
        aggr_mask_l3 = torch.stack(aggr_mask_l3, dim=1)

        aggr_fea = [aggr_fea_l1, aggr_fea_l2, aggr_fea_l3]
        aggr_mask = [aggr_mask_l1, aggr_mask_l2, aggr_mask_l3]
        out = []
        for i in range(len(aggr_fea)):
            out.append(self.fuse(aggr_fea[i]))
        
        recur_img = self.trans_conv(out[0][:, 1, :, :, :])

        return out, recur_img, aggr_mask

class Network(nn.Module):
    def __init__(self, in_chl=3, nf=64, front_blk=5, nbr=4, n_group=8, kernels=[3, 3, 3, 3], patches=[7, 11, 15], cor_ksize=3, n_times=4):
        super(Network, self).__init__()
        self.n_times = n_times

        self.first_conv = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        block = functools.partial(ResidualBlock_noBN_noAct, nf=nf)
        self.fea_ext = make_layer(block, front_blk)

        self.L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.recura1 = Recura(nf=nf, n_view=3, from_align=False)
        self.recurb1 = Recurb(nf=nf, n_view=3, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches, cor_ksize=cor_ksize)

        self.recura2 = Recura(nf=nf, n_view=3, from_align=True)
        self.recurb2 = Recurb(nf=nf, n_view=3, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches, cor_ksize=cor_ksize)

        self.recura3 = Recura(nf=nf, n_view=3, from_align=True)
        self.recurb3 = Recurb(nf=nf, n_view=3, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches, cor_ksize=cor_ksize)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, low_views):
        B, N, C, H, W = low_views.size()

        # feature extraction
        L1_fea = self.lrelu(self.first_conv(low_views.view(-1, C, H, W)))
        L1_fea = self.fea_ext(L1_fea)

        L2_fea = self.lrelu(self.L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.L2_conv2(L2_fea))

        L3_fea = self.lrelu(self.L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #
        fea, out1_a = self.recura1([L1_fea, L2_fea, L3_fea])
        fea, out1_b, mask = self.recurb1(fea, out1_a)

        fea, out2_a = self.recura2(fea, mask)
        fea, out2_b, mask = self.recurb2(fea, out2_a)

        fea, out3_a = self.recura3(fea, mask)
        fea, out3_b, mask = self.recurb3(fea, out3_a)
        
        return out1_a, out1_b, out2_a, out2_b, out3_a, out3_b
        # return out1, out2, out3


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Network(in_chl=3, nf=64, front_blk=5).to(device)
    input = torch.randn((2, 3, 3, 256, 256)).to(device)
    with torch.no_grad():
        out = net(input)

    print(True)
