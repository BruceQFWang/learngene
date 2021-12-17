from __future__ import print_function
import numpy as np
import time
import random
from PIL import Image
import os
import shutil
import errno
import sys
sys.path.append("../")
import csv
from pdb import set_trace as breakpoint
from matplotlib import pyplot as plt

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


_CIFAR_DATASET_DIR = './dataset/Cifar/cifar100'

_IMAGENET_DATASET_DIR = './dataset/Imagenet/Imagenet2012/Data/CLS-LOC'

time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

subtask_all_id = {}


def buildLabelIndex(labels):
    label2inds = {}
    
    for idx, label in enumerate(labels):
        
        if label not in label2inds:
            label2inds[label] = []
        
        label2inds[label].append(idx)

    return label2inds


class Denormalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        
        for t, m, s in zip(tensor, self.mean, self.std):
            
            t.mul_(s).add_(m) 
            
        return tensor


        
class DataLoader(object):
    
    def __init__(self, dataset, batch_size=1, task_iter=0, epoch_size=None, num_workers=0, shuffle=True):
        
        self.dataset = dataset
        
        self.shuffle = shuffle
        
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        
        self.batch_size = batch_size
        
        self.num_workers = num_workers

        mean_pix  = [0.485, 0.456, 0.406]
        std_pix   = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

        
    def get_iterator(self, epoch=0):
        
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
           
        def _load_function(idx):
            idx = idx % len(self.dataset)
            img, categorical_label = self.dataset[idx]
            img = self.transform(img)
            return img, categorical_label

        _collate_fun = default_collate
        
        
        tnt_dataset = tnt.dataset.ListDataset(elem_list = range(self.epoch_size), load =_load_function)
        
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size, collate_fn=_collate_fun, num_workers=self.num_workers, shuffle=self.shuffle)
        
        return data_loader

    def __call__(self, epoch=0): 
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
    
def get_permute_cifar100(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cate, path):
    
    dataloader_train = {}
    
    for i in range(num_tasks):
        
        task_folder_path = os.path.join(path, 'Task_' + str(i))
        
        transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                    
        dataset_train = datasets.ImageFolder(task_folder_path, transforms.Compose(transforms_list))
        
        dataloader_train[i] = DataLoader(dataset_train, batch_size, task_iter=i, epoch_size = None, num_workers = 0, shuffle = True)

    return dataloader_train

# training/test
def get_inheritable_cifar100(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cate, path):
    
    dataloader_train = {}
    dataloader_test = {}
    
    for i in range(num_tasks):
        
        task_folder_path = os.path.join( path, 'Task_' + str(i),'inheritable_traindata_' + str(num_imgs_per_cate) )
        
        transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                    
        dataset_train = datasets.ImageFolder(task_folder_path, transforms.Compose(transforms_list))
        
        dataloader_train[i] = DataLoader(dataset_train, batch_size, task_iter=i, epoch_size = None, num_workers = 0, shuffle = True)

    for i in range(num_tasks):
        
        task_folder_path = os.path.join( path, 'Task_' + str(i),'inheritable_testdata_50_' + str(num_imgs_per_cate) )
        
        transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                    
        dataset_test = datasets.ImageFolder(task_folder_path, transforms.Compose(transforms_list))
        
        dataloader_test[i] = DataLoader(dataset_test, batch_size, task_iter=i, epoch_size = None, num_workers = 0, shuffle = True)

    return dataloader_train, dataloader_test