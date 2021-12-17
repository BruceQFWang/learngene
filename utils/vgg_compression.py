from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils
import os

cfg = {
    'A' : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 512, 512, 'M'],
    'B' : [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'C' : [32, 'M', 64, 'M', 128, 'M', 256, 'M',256,256,256,'M']
}

class vgg_compression(nn.Module):
    def __init__(self, layers, num_class=100):
        super().__init__()
        self.features = make_layers(cfg['B'], batch_norm=True)
        self.layers = layers

        for layer in self.layers:
            self.features.append(layer)

        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        
        self.classifier = nn.ModuleList([
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            #nn.ReLU(inplace=False),
            #nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            #nn.ReLU(inplace=False),
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
    
    

def make_layers(cfg, batch_norm):
        #layers = []
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
        #layers.append(nn.ReLU(inplace=False))
        input_channel = l

    return layers  

