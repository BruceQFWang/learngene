import argparse
import time
import datetime
import sys
sys.path.append('../')
import copy
import numpy as np
import os
import shutil
import random
import warnings
import xlwt
import dill as pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable

from utils.train import train, test, train_ewc, test_ewc, train_ewc_vgg, test_ewc_vgg

from utils.model_zoo_cifar100 import vgg_compression_ONE
# from utils.imagenetDataloader import getDataloader_imagenet_inheritable
from utils.network_wider_cifar100 import Netwider
from utils.cifar100_dataloader import get_permute_cifar100, get_inheritable_cifar100

torch.cuda.set_device(0)  

parser = argparse.ArgumentParser(description='i_inheritable_random_trainnum200_wEWC')

parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--num_works', type=int, default=21, help='number of tasks')

parser.add_argument('--num_works_tt', type=int, default=10, help='number of comp_tasks')

parser.add_argument('--lr_drop', type=float, default=0.4)

parser.add_argument('--epochs_drop', type=int, default=110)

parser.add_argument('--print_freq', type=int, default=50)

parser.add_argument('--num_imgs_per_cat_train', type=int, default=20)

parser.add_argument('--path', type=str, default='./', help='path of base classes')


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
lr_ =args.lr

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

record_time = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H:%M:%S'))

RESULT_PATH_VAL = ''


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print('making dir: %s'%output_dir)
        
    torch.save(states, os.path.join(output_dir, filename))
    
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


def main():
    
    print("Data loading...")
    
    # trainloader_inheritable, testloader_inheritable = getDataloader_imagenet_inheritable(args.num_works_tt, args.batch_size, subtask_classes_num=5, num_imgs_per_cate=200)
    
    # cifar100
    inherit_path = args.path
    trainloader_inheritable, testloader_inheritable = get_inheritable_cifar100(args.num_works_tt, args.batch_size, subtask_classes_num=5, num_imgs_per_cate=args.num_imgs_per_cat_train, path=inherit_path)

    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    
    print("Model constructing...")
    model = Netwider(13)
    model.printf()
    
    for task in range(args.num_works):
        
        print("LL_Task {} begins !".format(task))
        
        start = time.time() 
        
        if task != 0:
            model_ = copy.deepcopy(model)
            del model
            model = model_
            model.wider(task-1)
            model.printf()
            
        if args.cuda:
            model = model.cuda()
        
        
        args.lr = lr_
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_epoch = 0
        
        snapshot = './imagenet_exp/outputs/i_inheritable_cifar100_wEWC/record_{0}/Task_{1}'.format(record_time, task)
        snapshot_model = './imagenet_exp/val_outputs/Val_lifelong_scratch_cifar/2021-08-24 01:28:41/task_{0}'.format(task) 
        # load collective-model
        
        if not os.path.isdir(snapshot):
            print("Building snapshot file: {0}".format(snapshot))
            os.makedirs(snapshot)
            
        checkpoint_path = os.path.join(snapshot_model, 'checkpoint.pth')  
        
        if os.path.isfile(checkpoint_path):
            print("loading success")
            checkpoint = torch.load(checkpoint_path)
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            
        train_name = 'LL_Train_'+str(task)
        test_name = 'LL_Test_'+str(task)
            
        print("LL_Task {0} finished ! ".format(task))
        
        if task == 20:
            
            layers = model.get_layers_19_20()
            
            model_vgg = vgg_compression_ONE(layers, 2048, num_class = 5)
            model_v = copy.deepcopy(model_vgg)
            del model_vgg
            model_vgg = model_v
            # model_vgg.printf()
            model_vgg = model_vgg.cuda()
            
            for task_tt in range(args.num_works_tt):
                
                print("TT_Task {} begins !".format(task_tt))

                train_tt_name = 'TT_Train_200_' + str(task_tt)
                test_tt_name = 'TT_Test_200_' + str(task_tt)
                
                sheet_task = book.add_sheet('TT_Task_200_{0}'.format(task_tt), cell_overwrite_ok=True)
                cnt_epoch = 0
        
                for epoch in range(args.epochs):
                    
                    optimizer = optim.SGD(model_vgg.parameters(), lr=args.lr, momentum=args.momentum,  weight_decay=5e-4)
                    
                    criterion = nn.CrossEntropyLoss()
                    
                    train_loss, train_acc = train_ewc(trainloader_inheritable[task_tt](epoch), model_vgg, criterion, optimizer, epoch, args, \
                                                  snapshot=snapshot, name=train_tt_name)

                    test_loss, test_acc = test_ewc(testloader_inheritable[task_tt](epoch), model_vgg, criterion, optimizer, epoch, args, \
                                               snapshot=snapshot, name=test_tt_name)

                    sheet_task.write(cnt_epoch, 0, 'Epoch_{0}'.format(cnt_epoch))
                    sheet_task.write(cnt_epoch, 1, train_loss)
                    sheet_task.write(cnt_epoch, 2, train_acc.item())
                    sheet_task.write(cnt_epoch, 3, test_loss)
                    sheet_task.write(cnt_epoch, 4, test_acc.item())     
                
                    cnt_epoch = cnt_epoch + 1
                    
                del model_vgg
                model_vgg = model_v
                # model_vgg.printf()
                model_vgg = model_vgg.cuda()
            
                print("TT_Task {0} finished !".format(task_tt))
                

    book.save(r'./imagenet_exp/outputs/i_inheritable_random_trainnum200_wEWC/i_inheritable_cifar100_wEWC.xls')

    
if __name__ == "__main__":
    main()

