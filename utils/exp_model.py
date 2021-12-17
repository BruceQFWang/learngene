from functools import reduce
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable

import utils

cfg = [ [64, 'M', 128, 'M', 128, 128, 128, 'M'], 
      [64, 'M', 128, 'M', 160, 128, 128, 'M'],
      [64, 'M', 128, 'M', 192, 128, 128, 'M'],
      [64, 'M', 128, 'M', 224, 128, 128, 'M'],
      [64, 'M', 128, 'M', 256, 128, 128, 'M'],
      [64, 'M', 128, 'M', 128, 160, 128, 'M'],
      [64, 'M', 128, 'M', 128, 192, 128, 'M'],
      [64, 'M', 128, 'M', 128, 224, 128, 'M'],
      [64, 'M', 128, 'M', 128, 256, 128, 'M'],
    ]

class expModel(nn.Module):
    def __init__(self, index, num_class=100):
        super().__init__()  
        
        cfg[0][-2] = 128 + (index + 1) * 32
        cfg_n = cfg[0]
        # print(cfg_n)
        
        self.features = make_layers(cfg_n, batch_norm=True)
        
        batch_size = 16
        
        self.classifier = nn.ModuleList([
            nn.Linear((128 + (index + 1) * 32) * batch_size, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(2048, num_class)
        ])

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(x.size()[0], -1)
        for layer in self.classifier:
            x = layer(x)
        return x
    
    def printf(self):
        for i, layer in enumerate(self.features):
            print("{}-layer:{}".format(i, layer))
        for i, layer in enumerate(self.classifier):
            print("{}-layer:{}".format(i, layer))

            
def make_layers(cfg, batch_norm):

    layers = nn.ModuleList([])

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(input_channel, l, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(l))

        layers.append(nn.ReLU(inplace=True))
        input_channel = l
        
    return layers  


class expModel_one(nn.Module):
    def __init__(self, index, num_class=100):
        super().__init__()  
        
        self.features = make_layers(cfg[index], batch_norm=True)
    
        batch_size = 16
        self.classifier = nn.ModuleList([
            nn.Linear(128 * batch_size, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(2048, num_class)
        ])

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(x.size()[0], -1)
        for layer in self.classifier:
            x = layer(x)
        return x
    
    def printf(self):
        for i, layer in enumerate(self.features):
            print("{}-layer:{}".format(i,layer))
        for i, layer in enumerate(self.classifier):
            print("{}-layer:{}".format(i,layer))
    