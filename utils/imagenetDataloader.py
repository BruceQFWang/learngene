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
    
    
def getDataloader_imagenet_continual(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cate):
    
    dataloader_train = {}
    
    for i in range(num_tasks):
        
        task_folder_path = os.path.join('../utils/exp_data/data_imagenet/2021-12-12_00:02:32/continualdataset', 'Task_' + str(i))
        
        transforms_list = [
                        transforms.Resize(84),
                        transforms.RandomCrop(84),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                    
        dataset_train = datasets.ImageFolder(task_folder_path, transforms.Compose(transforms_list))
        
        dataloader_train[i] = DataLoader(dataset_train, batch_size, task_iter=i, epoch_size = None, num_workers = 0, shuffle = True)

    return dataloader_train


def getDataloader_imagenet_inheritable(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cate, path):
    
    dataloader_train = {}
    dataloader_test = {}

    for i in range(num_tasks):
        
        task_folder_path = os.path.join( path, 'Task_' + str(i),'inheritable_traindata_' + str(num_imgs_per_cate) )
        #task_folder_path = os.path.join('./utils/exp_data/data_imagenet/2021-12-12_00:02:32/inheritabledataset', \
                                        'Task_' + str(i),'inheritable_traindata_' + str(num_imgs_per_cate))
        
        transforms_list = [
                        transforms.Resize(84),
                        transforms.RandomCrop(84),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                    
        dataset_train = datasets.ImageFolder(task_folder_path, transforms.Compose(transforms_list))
        
        dataloader_train[i] = DataLoader(dataset_train, batch_size, task_iter=i, epoch_size = None, num_workers = 0, shuffle = True)

        
    for i in range(num_tasks):
        
        task_folder_path = os.path.join( path, 'Task_' + str(i),'inheritable_testdata_50_' + str(num_imgs_per_cate) )
        #task_folder_path = os.path.join('./utils/exp_data/data_imagenet/2021-12-12_00:02:32/inheritabledataset', \
                                        'Task_' + str(i),'inheritable_testdata_50_' + str(num_imgs_per_cate))
        
        transforms_list = [
                        transforms.Resize(84),
                        transforms.RandomCrop(84),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                    
        dataset_test = datasets.ImageFolder(task_folder_path, transforms.Compose(transforms_list))
        
        dataloader_test[i] = DataLoader(dataset_test, batch_size, task_iter=i, epoch_size = None, num_workers = 0, shuffle = True)
        
    return dataloader_train, dataloader_test



#     trainloader_continual = getDataloader_imagenet_continual(num_tasks, batch_size, subtask_classes_num=5, num_imgs_per_cate=400)

#     trainloader_inheritable, testloader_inheritable = getDataloader_imagenet_inheritable(num_tasks, batch_size, subtask_classes_num=5, num_imgs_per_cate=100)
