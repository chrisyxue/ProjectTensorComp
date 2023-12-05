# -*- coding: UTF-8 -*-

""" Difference between this model and usual resnet
The first conv kernel is 3 and stride is 1, while another is 7 and stride is 2.
"""

from typing import Union

import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tednet.tnn.tucker2.base import TK2Conv2D, TK2Linear
from tednet.tnn.tn_module import LambdaLayer
# from .base import TK2Conv2D, TK2Linear
# from ..tn_module import LambdaLayer



class TKPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, r, stride=1, **kwargs):
        super(TKPreActBlock, self).__init__()
        c_in = in_planes
        c_out = planes

        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = TK2Conv2D(c_in, c_out, [r, r],  kernel_size=3, padding=1,
                              stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = TK2Conv2D(c_out, c_out, [r, r], kernel_size=3,
                              stride=1,
                              padding=1, bias=False)
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                     TK2Conv2D(c_in, c_out, [r, r],
                              kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class TKPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, ranks, num_classes=10, init_channels=64):
        super(TKPreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels
        assert(len(ranks) == 5)
        self.conv1 = TK2Conv2D(3, c, [ranks[0],ranks[0]], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1, r=ranks[1])
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2, r=ranks[2])
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2, r=ranks[3])
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2, r=ranks[4])
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

        # self.linear = TK2Linear([4, 4, 4], num_classes, [r_list[5], r_list[5]], bias=True)

    def _make_layer(self, block, planes, num_blocks, stride, r):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, r=r))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

def make_tkresnet18k(k=64, num_classes=10, rank_ratio=0.5) -> TKPreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    k_compress = int(rank_ratio * k)
    ranks = [2,k_compress,2*k_compress,4*k_compress,8*k_compress]
    return TKPreActResNet(TKPreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k, ranks=ranks)
