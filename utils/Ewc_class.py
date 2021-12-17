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
from torch import autograd
from torch.autograd import Variable

class EWC(object):
    
    def __init__(self, model: nn.Module, cuda, lamda=40):
        self.model = model
        #self.dataset = dataset
        self._is_on_cuda = cuda
        self.lamda = lamda
        
    def estimate_fisher(self, data_loader, sample_size, batch_size=32):
        
        loglikelihoods = []
        for x, y in data_loader:
            
            x = Variable(x).cuda() if self._is_on_cuda else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda else Variable(y)
            

            loglikelihoods.append(F.log_softmax(self.model(x), dim=1)[range(batch_size), y.data])

            if len(loglikelihoods) >= 1: 
                break
          
        print("Estimate the fisher information of the parameters:\n one")
    
        loglikelihoods = torch.cat(loglikelihoods).unbind()

        loglikelihood_grads = zip(*[autograd.grad(l, self.model.parameters(), retain_graph=(i < len(loglikelihoods)), allow_unused=True) \
                                    for i, l in enumerate(loglikelihoods, 1)])

        print("next:")
        
        fisher_diagonals = []
        os.system("nvidia-smi")

        for gs in loglikelihood_grads:
            try:
                gs = torch.stack(gs)
                g = 0
                for _b in range(gs.shape[0]):
                    g += (gs[_b]).pow(2)
                g /= gs.shape[0]
                fisher_diagonals.append(g)
            except:
                print("========> Why am i here? <========")
                continue
            torch.cuda.empty_cache()
            
        param_names = [n.replace('.', '__') for n, p in self.model.named_parameters()]
        
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            try:
                self.model.register_buffer('{}_mean'.format(n), p.data.clone())
                self.model.register_buffer('{}_fisher'.format(n), fisher[n].data.clone())
            except:
                continue
                
    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for index, (n, p) in enumerate(self.model.named_parameters()):
                # retrieve the consolidated mean and fisher information.
                
                if index<16 or index>27:  
                    continue
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            print("ewc loss > 0")
            return (self.lamda/2)*sum(losses)
        
        except AttributeError:
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

class EWC_vgg(object):
    def __init__(self, model: nn.Module, cuda, lamda=40 ):
        self.model = model
        
        self._is_on_cuda = cuda
        self.lamda = lamda
        
    def estimate_fisher(self, data_loader, sample_size, batch_size=32):

        loglikelihoods = []
        for x, y in data_loader:
            x = Variable(x).cuda() if self._is_on_cuda else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda else Variable(y)

            loglikelihoods.append(
                F.log_softmax(self.model(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break

        print("Estimate the fisher information of the parameters:\n one")
        loglikelihoods = torch.cat(loglikelihoods).unbind()

        loglikelihood_grads = zip(*[autograd.grad(
            l, self.model.parameters(),
            retain_graph=(i < len(loglikelihoods)),
            allow_unused=True
        ) for i, l in enumerate(loglikelihoods, 1)])

        print("next:")
        fisher_diagonals = []
        os.system("nvidia-smi")

        for gs in loglikelihood_grads:
            try:
                gs = torch.stack(gs)
                g = 0
                for _b in range(gs.shape[0]):
                    g += (gs[_b]).pow(2)
                g /= gs.shape[0]
                fisher_diagonals.append(g)
            except:
                print("========> Why am i here? <========")
                continue
            torch.cuda.empty_cache()
        param_names = [
            n.replace('.', '__') for n, p in self.model.named_parameters()
        ]
        
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            try:
                self.model.register_buffer('{}_mean'.format(n), p.data.clone())
                self.model.register_buffer('{}_fisher'.format(n), fisher[n].data.clone())
            except:
                continue
                
    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for index, (n, p) in enumerate(self.model.named_parameters()):

                if index<34 or index>45:
                    continue
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                
                mean = Variable(mean)
                fisher = Variable(fisher)
                losses.append((fisher * (p-mean)**2).sum())
            print("ewc loss > 0")
            return (self.lamda/2)*sum(losses)
        
        except AttributeError:
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )
        