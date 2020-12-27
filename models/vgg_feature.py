from PIL import Image
from .vgg import Vgg19
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


path_A = '../dataset/trainA/312.png'
path_B = '../dataset/trainB/312.png'

self.vgg = Vgg19().cuda()

img_A = Image.open(path_A).convert('RGB')
img_B = Image.open(path_B).convert('RGB')



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