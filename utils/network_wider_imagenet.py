import argparse
import time
import sys
sys.path.append('../')
import copy
import numpy as np
import os
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch import autograd

cfg_m = [64, 64,    64, 64,     128, 128, 128,    256, 256, 256,    256, 256, 256]

m =[1, 3, 6, 9, 12]

add_width = [64,64, 64,64, 64,64, 64,64, 128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
add_conv =  [2,3,   4,4,  5,5,  6,6,   7,7,   8,8,    9,9,   10,10,  11,11,  12,12 ]

class Netwider(nn.Module):
    
    def __init__(self, num_conv, num_classes=100):
        
        super(Netwider, self).__init__()
        self.num_class = num_classes
        self.num_conv = num_conv
        
        self.layers, self.num_pooling, self.conv_idx, self.output_channel = make_layers(self.num_conv, batch_norm=True)
        
       
        image_length = 84
        image_width = 84
        self.classifier = make_classifier(self.output_channel*(int(image_length/(2**self.num_pooling)))*(int(image_width/(2**self.num_pooling))),\
                                          self.num_class)

    def forward(self, x):
        for i, layer in enumerate(self.layers) :
            x = layer(x)
        x = x.view(x.size()[0], -1)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
        return x
    
    def printf(self):
        for i, layer in enumerate(self.layers):
            print("{}-layer:{}".format(i,layer)) 
        for i, layer in enumerate(self.classifier):
            print("{}-layer:{}".format(i,layer))

    def wider(self, task):
        if add_conv[task] != 12:
            self.layers[self.conv_idx[add_conv[task]]], \
            self.layers[self.conv_idx[add_conv[task]+1]],\
            self.layers[self.conv_idx[add_conv[task]]+1] = \
            make_wider(self.layers[self.conv_idx[add_conv[task]]],\
                   self.layers[self.conv_idx[add_conv[task]+1]], \
                   add_width[task], \
                   self.layers[self.conv_idx[add_conv[task]]+1])
        else:
            self.layers[self.conv_idx[add_conv[task]]],\
            self.classifier[0],\
            self.layers[self.conv_idx[add_conv[task]]+1] = \
            make_wider(self.layers[self.conv_idx[add_conv[task]]],\
                   self.classifier[0], \
                   add_width[task], \
                   self.layers[self.conv_idx[add_conv[task]]+1])
    
        
    def get_layers_5_6(self):
        layer_f = self.layers[10:13]
        layer_l = self.layers[14:20]
        for layer in layer_l:
            layer_f.append(layer)
        return layer_f
    
    def get_layers_7_8(self):
        return self.layers[14:23]
    
    def get_layers_9_10(self):
        layer_f = self.layers[17:23]
        layer_l = self.layers[24:27]
        for layer in layer_l:
            layer_f.append(layer)
        return layer_f

    def get_layers_11_12(self):
        layer_f = self.layers[20:23]
        layer_l = self.layers[24:30]
        for layer in layer_l:
            layer_f.append(layer)
        return layer_f

    def get_layers_13_14(self):
        return self.layers[24:33]

    def get_layers_15_16(self):
        layer_f = self.layers[27:33]
        layer_l = self.layers[34:37]
        for layer in layer_l:
            layer_f.append(layer)
        return layer_f

    def get_layers_17_18(self):
        layer_f = self.layers[30:33]
        layer_l = self.layers[34:40]
        for layer in layer_l:
            layer_f.append(layer)
        return layer_f

    def get_layers_19_20(self):
        return self.layers[-10:-1]
    
    def get_convlayers(self):
        return self.layers
    
    def get_all(self):
        return self.layers, self.classifier
    
def make_layers(num_conv, batch_norm):
 
    layers = nn.ModuleList([])
    num_pooling = 0
    layer_num = -1
    conv_indx = []
    cfg = cfg_m[ :num_conv]

    input_channel = 3
    for l in range(len(cfg)):
        # print(cfg)
        
        # Conv
        layers.append(nn.Conv2d(input_channel, cfg_m[l], kernel_size=3, padding=1))
        layer_num += 1
        conv_indx.append(layer_num)
        
        # BN
        if batch_norm:
            layers.append(nn.BatchNorm2d(cfg_m[l]))
            layer_num += 1
        
        # ReLU
        layers.append(nn.ReLU(inplace=True))
        layer_num += 1
        
        # MaxPooling
        if l in m:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            num_pooling += 1
            layer_num += 1
            continue
            
        input_channel = cfg_m[l]

    return layers, num_pooling, conv_indx, cfg[-1]


def make_classifier(in_channel,num_class):
    classifier = nn.ModuleList([
            nn.Linear(in_channel , 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_class, bias=True),
            nn.Softmax()
        ])
    
    return classifier


def make_wider(m1, m2, add_width, bnorm=None, out_size=None, noise=False, random_init=False, weight_norm=False):
    

    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data
    old_width = w1.size(0)
    new_width = old_width + add_width

    if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1)//factor**2, factor, factor)
            elif w1.dim() == 5:
                assert out_size is not None,\
                       "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(w2.size(0), w2.size(1)//factor, out_size[0],
                             out_size[1], out_size[2])
        else:
            assert w1.size(0) == w2.size(1), "Module weights are not compatible"
        assert new_width > w1.size(0), "New size should be larger"

        
        nw1 = m1.weight.data.clone()
        nw2 = w2.clone()

        if nw1.dim() == 4:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))
        elif nw1.dim() == 5:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3), nw1.size(4))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3), nw2.size(4))
        else:
            nw1.resize_(new_width, nw1.size(1))
            nw2.resize_(nw2.size(0), new_width)

        if b1 is not None:
            nb1 = m1.bias.data.clone()
            nb1.resize_(new_width)

        if bnorm is not None:
            nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
            nrunning_var = bnorm.running_var.clone().resize_(new_width)
            if bnorm.affine:
                nweight = bnorm.weight.data.clone().resize_(new_width)
                nbias = bnorm.bias.data.clone().resize_(new_width)

        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)

        nw1.narrow(0, 0, old_width).copy_(w1)
        nw2.narrow(0, 0, old_width).copy_(w2)
        nb1.narrow(0, 0, old_width).copy_(b1)

        if bnorm is not None:
            nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
            nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
            if bnorm.affine:
                nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

        # TEST:normalize weights
        if weight_norm:
            for i in range(old_width):
                norm = w1.select(0, i).norm()
                w1.select(0, i).div_(norm)

        # select weights randomly
        tracking = dict()
        for i in range(old_width, new_width):
            idx = np.random.randint(0, old_width)
            try:
                tracking[idx].append(i)
            except:
                tracking[idx] = [idx]
                tracking[idx].append(i)

            # TEST:random init for new units
            if random_init:
                n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                if m2.weight.dim() == 4:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                elif m2.weight.dim() == 5:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                elif m2.weight.dim() == 2:
                    n2 = m2.out_features * m2.in_features
                nw1.select(0, i).normal_(0, np.sqrt(2./n))
                nw2.select(0, i).normal_(0, np.sqrt(2./n2))
            else:
                nw1.select(0, i).copy_(w1.select(0, idx).clone())
                nw2.select(0, i).copy_(w2.select(0, idx).clone())
            nb1[i] = b1[idx]

        if bnorm is not None:
            nrunning_mean[i] = bnorm.running_mean[idx]
            nrunning_var[i] = bnorm.running_var[idx]
            if bnorm.affine:
                nweight[i] = bnorm.weight.data[idx]
                nbias[i] = bnorm.bias.data[idx]
            bnorm.num_features = new_width

        if not random_init:
            for idx, d in tracking.items():
                for item in d:
                    nw2[item].div_(len(d))

        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m1.out_channels = new_width
        m2.in_channels = new_width

        if noise:
            noise = np.random.normal(scale=5e-2 * nw1.std(),
                                     size=list(nw1.size()))
            nw1 += torch.FloatTensor(noise).type_as(nw1)

        m1.weight.data = nw1

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            if w1.dim() == 4:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor**2)
                m2.in_features = new_width*factor**2
            elif w2.dim() == 5:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor)
                m2.in_features = new_width*factor
        else:
            m2.weight.data = nw2

        m1.bias.data = nb1

        if bnorm is not None:
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias
        return m1, m2, bnorm
    