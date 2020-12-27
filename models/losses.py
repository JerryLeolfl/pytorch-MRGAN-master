import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
from torch import autograd
import torchvision.models as models
from .vgg import Vgg19

###############################################################################
# Functions
###############################################################################


class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return h_tv/count_h + w_tv/count_w

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())*0.1
        
        return loss
