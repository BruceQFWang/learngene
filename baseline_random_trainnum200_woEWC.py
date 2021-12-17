import argparse
import time
import datetime
import sys
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

from utils.network_wider_imagenet import Netwider
from utils.train import train, test, train_ewc, test_ewc, train_ewc_vgg, test_ewc_vgg
from utils.model_zoo_imagenet import network_random, network_vgg
from utils.imagenetDataloader import getDataloader_imagenet_inheritable

parser = argparse.ArgumentParser(description='i_baseline_random_trainnum200_woEWC')

parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')

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

parser.add_argument('--retrain', action='store_true', default=False, help='retrain the model')


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
    trainloader_inheritable, testloader_inheritable = getDataloader_imagenet_inheritable(args.num_works_tt, args.batch_size, subtask_classes_num=5, num_imgs_per_cate=200)
 
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    
    for task in range(args.num_works_tt):
        
        print("TT_Task {} begins !".format(task))
        
        start = time.time()   
        model = network_random(num_class=5)
        
        if args.cuda:
            model = model.cuda()
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_epoch = 0
        
        snapshot = './outputs_baseline/i_baseline_random_trainnum200_woEWC/record_{0}/Task_{1}'.format(record_time, task)
        
        if not os.path.isdir(snapshot):
            print("Building snapshot file: {0}".format(snapshot))
            os.makedirs(snapshot)
            
        checkpoint_path = os.path.join(snapshot, 'checkpoint.pth')
        
        train_name = 'TT_Train_'+str(task)
        test_name = 'TT_Test_'+str(task)
        
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])

        
        sheet_task = book.add_sheet('TT_Task_200_{0}'.format(task), cell_overwrite_ok=True)
        cnt_epoch = 0
        
        for epoch in range(args.epochs):
            
            start = time.time()
            
            is_best = False
            
            train_loss, train_acc = train(trainloader_inheritable[task](epoch), model, criterion, optimizer, epoch, args, snapshot=snapshot, \
                                          name=train_name, RESULT_PATH_VAL=RESULT_PATH_VAL)
            
            test_loss, test_acc = test(testloader_inheritable[task](epoch), model, criterion, optimizer, epoch, args, snapshot=snapshot, \
                                       name=test_name, RESULT_PATH_VAL=RESULT_PATH_VAL)
            
            sheet_task.write(cnt_epoch, 0, 'Epoch_{0}'.format(cnt_epoch))
            sheet_task.write(cnt_epoch, 1, train_loss)
            sheet_task.write(cnt_epoch, 2, train_acc.item())
            sheet_task.write(cnt_epoch, 3, test_loss)
            sheet_task.write(cnt_epoch, 4, test_acc.item())
            
            cnt_epoch = cnt_epoch + 1
             
            print('Training: Average loss: {:.4f}, Accuracy: {:.4f}'.format(train_loss, train_acc))
            print('Testing: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
            
            '''save_checkpoint({
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                #'best_state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'train_acc': train_acc,
            }, is_best=is_best, output_dir=snapshot)
            
            # save best model
            if test_acc > best_acc:
                is_best = True
                best_acc = test_acc
                save_checkpoint({
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    #'best_state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                }, is_best=is_best, output_dir=snapshot)'''
            
            finish = time.time()
            
            print('Epoch {} training and testing time consumed: {:.2f}s'.format(epoch, finish - start))
            
        print("TT_Task {} finished ! ".format(task))
        
    book.save(r'./outputs_baseline/i_baseline_random_trainnum200_woEWC/record_{0}/i_baseline_random_trainnum200_woEWC.xls'\
              .format(record_time))

    
if __name__ == "__main__":
    main()

