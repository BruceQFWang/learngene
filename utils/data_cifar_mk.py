from __future__ import print_function
import numpy as np
import dill as pickle
import time
import datetime
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
import argparse

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

parser = argparse.ArgumentParser(description='data_cifar_mk')

parser.add_argument('--num_imgs_per_cat_train', type=int, default=200)
parser.add_argument('--path', type=str, default='./', help='path of base classes')

args = parser.parse_args()


_CIFAR_DATASET_DIR = args.path #Cifar100 datapath


_IMAGENET_DATASET_DIR = './dataset/Imagenet/Imagenet2012/Data/CLS-LOC'

time_now = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H:%M:%S'))

subtask_all_id = {}

class Denormalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        
        for t, m, s in zip(tensor, self.mean, self.std):
            
            t.mul_(s).add_(m) 
            
        return tensor


class GenericDataset_ll(data.Dataset):
    
    def __init__(self, dir_name, dataset_name, split, task_iter_item = 0, subtask_class_num=5, \
                 random_sized_crop=False, num_imgs_per_cat=300, label_division=True):
        
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat
  
        global subtask_all_id
        global time_now
        
        if self.dataset_name=='cifar100':
        
            self.mean_pix = CIFAR100_TRAIN_MEAN
            self.std_pix = CIFAR100_TRAIN_STD

        
            if self.split != 'train':
                transforms_list = [
                    transforms.Resize(32), 
                    transforms.CenterCrop(32),
                    lambda x: np.asarray(x), 
                ]
            
            else:
                if self.random_sized_crop:
                    transforms_list = [   
                        transforms.RandomResizedCrop(32),             
                        transforms.RandomHorizontalFlip(),     
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]
                    
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _CIFAR_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform) 
            
            
            subtask_all_id[task_iter_item] = random.sample(range(0,64), subtask_class_num)
            
            classes = [k for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]
            
            class_to_idx = {k : v for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}
            
            imgs = []
            
            task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'continualdataset', \
                               'Task_' + str(task_iter_item))
            
            if not os.path.exists(task_folder_path):
                os.makedirs(task_folder_path)
            
            
            with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'), 'a') as file_val:
                file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, 'continualdata', class_to_idx))
                
                
            for c in classes:
                
                pths = os.path.join(_CIFAR_DATASET_DIR + '/' + self.split, c)
                
                task_class_folder_path = os.path.join(task_folder_path, c)
                
                if not os.path.exists(task_class_folder_path):
                    os.makedirs(task_class_folder_path)
                
                all_imgs = os.listdir(pths)   
                
                samples = all_imgs[0:num_imgs_per_cat]
                
                for sp in samples:
                    shutil.copy(os.path.join(_CIFAR_DATASET_DIR + '/' + self.split, c, sp), task_class_folder_path)
                    imgs.append((os.path.join(_CIFAR_DATASET_DIR + '/' + self.split, c, sp), class_to_idx[c]))
                
            self.data = datasets.ImageFolder(task_folder_path, self.transform)
            
            
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class GenericDataset_mm(data.Dataset):
    
    def __init__(self, dir_name, dataset_name, split, task_iter_item = 0, subtask_class_num=5, \
                 random_sized_crop=False, num_imgs_per_cat=300, label_division=True):
        
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat
  
        global subtask_all_id
        global time_now
        
        if self.dataset_name=='cifar100':
        
            self.mean_pix = CIFAR100_TRAIN_MEAN
            self.std_pix = CIFAR100_TRAIN_STD

        
            if self.split != 'train':
                transforms_list = [
                    transforms.Resize(32), 
                    transforms.CenterCrop(32),
                    lambda x: np.asarray(x), 
                ]
            
            else:
                if self.random_sized_crop:
                    transforms_list = [   
                        transforms.RandomResizedCrop(32),             
                        transforms.RandomHorizontalFlip(),     
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(32),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        lambda x: np.asarray(x),
                    ]
                    
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _CIFAR_DATASET_DIR + '/novel/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform) 
            
            global subtask_all_id
            global time_now
            if self.split == 'train':
                subtask_all_id[task_iter_item] = random.sample(range(0, 20), subtask_class_num)
                re = 'inheritable_traindata_' + str(num_imgs_per_cat)
            else:
                re = 'inheritable_testdata_50_' + str(num_imgs_per_cat)
            
            
            classes = [k for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]]
            
            class_to_idx = {k : v for (k,v) in self.data.class_to_idx.items() if v in subtask_all_id[task_iter_item]}
            
            imgs = []
            
            task_folder_path = os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'inheritabledataset', \
                               'Task_' + str(task_iter_item), re)
            
            if not os.path.exists(task_folder_path):
                os.makedirs(task_folder_path)

            with open(os.path.join('./exp_data/{0}'.format(dir_name), time_now, 'record_task_info'), 'a') as file_val:
                file_val.write('Task {0}:\n {1} : {2}\n'.format(task_iter_item, re, class_to_idx))    
                
            for c in classes:
                
                pths = os.path.join(_CIFAR_DATASET_DIR + '/novel/' + self.split, c)
                
                task_class_folder_path = os.path.join(task_folder_path, c)
                
                if not os.path.exists(task_class_folder_path):
                    os.makedirs(task_class_folder_path)
                
                all_imgs = os.listdir(pths)   
                
                if self.split == 'train':
                    num_sample = min(len(all_imgs), num_imgs_per_cat)
                    samples = random.sample(all_imgs[400:], num_sample)
                        
                else:
                    num_sample = len(all_imgs)
                    samples = random.sample(all_imgs, num_sample)
                
                for sp in samples:
                    shutil.copy(os.path.join(_CIFAR_DATASET_DIR + '/novel/' + self.split, c, sp), task_class_folder_path)
                    imgs.append((os.path.join(_CIFAR_DATASET_DIR + '/novel/' + self.split, c, sp), class_to_idx[c]))
                
            self.data = datasets.ImageFolder(task_folder_path, self.transform)
            
            
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

        
class DataLoader(object):
    
    def __init__(self, dataset, batch_size=1, task_iter=0, epoch_size=None, num_workers=0, shuffle=True):
        
        self.dataset = dataset
        
        self.shuffle = shuffle
        
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        
        self.batch_size = batch_size
        
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        
        
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

    
def get_cifar100_dataloaders(num_tasks, batch_size, subtask_classes_num, num_imgs_per_cat_train, num_imgs_per_cat_test, label_division):
    
    dataloader_train = {}
    dataloader_test = {}

    dataloader_train_mm = None

    dir_name = 'data_cifar100'
    
    # continual data
    for i in range(num_tasks):
        
        dataset_train = GenericDataset_ll(dir_name, 'cifar100','base', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                       random_sized_crop=False, num_imgs_per_cat = num_imgs_per_cat_train, label_division=label_division) 
        print('lifelong data Task {0} done! '.format(i))
    
    inh = [10, 20]
    
    
    # target data
    for i in range(0, num_tasks):
        
        for mm_vol in inh:
            dataset_train_mm = GenericDataset_mm(dir_name, 'cifar100','train', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                       random_sized_crop=False, num_imgs_per_cat = mm_vol, label_division=True)
        
            dataset_test = GenericDataset_mm(dir_name, 'cifar100','test', task_iter_item = i, subtask_class_num=subtask_classes_num, \
                      random_sized_crop=False, num_imgs_per_cat = mm_vol, label_division=True) 
        
        print('inheritable data Task {0} done! '.format(i))
        
    return dataset_train, dataloader_train_mm, dataloader_test
   

if __name__ == '__main__':

    args = parser.parse_args()
    
    num_tasks = 50
    
    num_epochs = 1
    
    batch_size = 64
    
    
    lifelong_train_loader = {}
    lifelong_test_loader = {}
    
    train_ll_loader, train_mm_loader, test_loader = \
        get_cifar100_dataloaders(
            num_tasks,
            batch_size, 
            subtask_classes_num=5, 
            num_imgs_per_cat_train=args.num_imgs_per_cat_train, 
            num_imgs_per_cat_test=100, 
            label_division=True)


