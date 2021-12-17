from functools import reduce
import utils
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable

cfg = {
    'A' : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 512, 512, 'M'],
    ##### 'B' : [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'B' : [64, 'M', 128, 'M', 256, 'M', 512],
    'C' : [64, 'M', 128, 'M', 128, 'M', 256, 'M'],
    'D' : [64, 'M', 64, 'M', 128, 'M', 128, 'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class vgg_compression_ONE(nn.Module):
    
    def __init__(self, layers, channel, num_class = 100):
        
        super().__init__()
        
        self.features = make_layers(cfg['B'], batch_norm=True)
        self.layers = layers

        for layer in self.layers:
            self.features.append(layer)

        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.ModuleList([
            nn.Linear(channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class)
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
            print("{0}-layer:{1}".format(i, layer))
        for i, layer in enumerate(self.classifier):
            print("{0}-layer:{1}".format(i, layer))

            
class vgg_compression_TWO(nn.Module):
    
    def __init__(self, layers, channel, num_class=100):
        
        super().__init__()
        self.features = make_layers(cfg['C'], batch_norm=True)
        self.layers = layers

        for layer in self.layers:
            self.features.append(layer)

        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.ModuleList([
            nn.Linear(channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class)
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
            print("{0}-layer:{1}".format(i, layer))
        for i, layer in enumerate(self.classifier):
            print("{0}-layer:{1}".format(i, layer))


class vgg_compression_THREE(nn.Module):
    
    def __init__(self, layers, channel, num_class=100):
        
        super().__init__()
        self.features = make_layers(cfg['D'], batch_norm=True)
        self.layers = layers

        for layer in self.layers:
            self.features.append(layer)

        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.ModuleList([
            nn.Linear(channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class)
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
            print("{0}-layer:{1}".format(i, layer))
        for i, layer in enumerate(self.classifier):
            print("{0}-layer:{1}".format(i, layer))
    
    
class network_random(nn.Module):
    
    def __init__(self, num_class=100):
        
        super().__init__()
        self.features = make_layers(cfg['A'], batch_norm=True)

        self.classifier = nn.ModuleList([
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class)
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
            print("{0}-layer:{1}".format(i, layer))
        for i, layer in enumerate(self.classifier):
            print("{0}-layer:{1}".format(i, layer))

            
class network_vgg(nn.Module):
    
    def __init__(self, num_class=100):
        
        super().__init__()
        self.features = make_layers(cfg['E'], batch_norm=True)

        self.classifier = nn.ModuleList([
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class)
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
            print("{0}-layer:{1}".format(i, layer))
        for i, layer in enumerate(self.classifier):
            print("{0}-layer:{1}".format(i, layer))
    
    
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

