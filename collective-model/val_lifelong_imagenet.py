import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
sys.path.append('../')
import copy
import numpy as np
import os
import shutil
import random
import xlwt

from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import warnings

from utils.data_imagenet_mk import get_permute_imagenet
from utils.imagenetDataloader import getDataloader_imagenet_continual

# from utils.ll_data_pre import get_imagenet_single_dataloader, get_cifar100_single_dataloader

from utils.network_wider_imagenet import Netwider

from utils.train import train, test, train_ewc, test_ewc

from utils.vgg_compression import vgg_compression

parser = argparse.ArgumentParser(description='PyTorch ImageNet')

parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')

parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')

parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')

parser.add_argument('--num_works', type=int, default=50, help='number of tasks')

parser.add_argument('--lr_drop', type=float, default=0.4)

parser.add_argument('--epochs_drop', type=int, default=110)

parser.add_argument('--print_freq', type=int, default=50)

parser.add_argument('--judgement', action='store_true', default=False,help='judgment criterion for inheriting learngene')

parser.add_argument('--path', type=str, default='./', help='path of base classes')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
lr_ =args.lr

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# global variable

RESULT_PATH_VAL = './val_save_results/' + 'imagenet_' + str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H:%M:%S'))

record_1_epoch = dict()

record_10_epoch = dict()

record_50_epoch = dict()

record_iter = 0


# save model
def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print('making dir: %s'%output_dir)
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))

        
# install hook_forward  
def hook_fn_forward(module, input, output):
    print(module) 
    print('input', input[0].shape) 
    print('output', output.shape)
    
    
# install hook_backward  
def hook_fn_backward(module, grad_input, grad_output):
    
    
    threshold1 = 5e-4
    lst_grad_in = list(grad_input)
    
    if len(lst_grad_in) > 1:
       
        # print('grad_input[1] size ', lst_grad_in[1].shape)
        # file.write('grad_input[1] size ' + str(lst_grad_in[1].shape) + '\n')
        num_input = 0
        ingra_shape = lst_grad_in[1].shape
        
        if len(ingra_shape) == 4:
            # print(module)
            n1 = ingra_shape[0]
            n2 = ingra_shape[1]
            n3 = ingra_shape[2]
            n4 = ingra_shape[3]
            
            num_input = np.where(grad_input[1].cuda().data.cpu().numpy() > threshold1)[0].shape[0]
            
            mul = n1 * n2 * n3 * n4
            print('the number of input gradients which are more than {0}: {1}, total size: {2}, rate: {3}.'.format(threshold1, num_input, mul, 1.0 * num_input/mul))
            
            record_1_epoch[module] = record_1_epoch[module] + 1.0 * num_input / mul
            record_10_epoch[module] = record_10_epoch[module] + 1.0 * num_input / mul
            record_50_epoch[module] = record_50_epoch[module] + 1.0 * num_input / mul
            

def main():
    
    print("Data loading...")
    with open(RESULT_PATH_VAL, 'a') as file_val:
        file_val.write('Data loading...\n\n')
    
    print()
    train_loader = getDataloader_imagenet_continual(args.num_works, args.batch_size, subtask_classes_num=5, num_imgs_per_cate=400)

  
    DIR_NAME = './val_outputs/exp_val'
    if not os.path.isdir(DIR_NAME):
        os.makedirs(DIR_NAME)
        
    ftrain_fn = os.path.join(DIR_NAME, 'wider_train_loss_tasks.txt')
    ftest_fn = os.path.join(DIR_NAME, 'wider_test_acc_tasks.txt')
    ftrain_acc_fn = os.path.join(DIR_NAME, 'wider_train_acc_tasks.txt')
    fdata_label = os.path.join(DIR_NAME,'label_wider.txt')
    
    ftrain= open(ftrain_fn,'a') 
    ftest = open(ftest_fn,'a') 
    ftrain_acc = open(ftrain_acc_fn,'a')
    
    # save train time
    localtime = time.localtime(time.time())
    ftrain.write(str(localtime)+'\n')
    ftest.write(str(localtime)+'\n')
    ftrain_acc.write(str(localtime)+'\n')
    
    param = 'epoch-{} learning rate-{} lr_drop-{} epochs_drop-{}'.format(args.epochs, args.lr, args.lr_drop, args.epochs_drop)
    ftrain.write(str(param)+'\n')
    ftest.write(str(param)+'\n')
    ftrain_acc.write(str(param)+'\n')

    print("Model loading...")
    print()
    with open(RESULT_PATH_VAL, 'a') as file_val:
        file_val.write('Model loading...\n\n')
    
    model = Netwider(13)
    # model.printf()
    modules = model.named_modules()
    if args.judgement:
        for name, module in modules:
            module.register_backward_hook(hook_fn_backward)
        
    
    now_time = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
    
    book=xlwt.Workbook(encoding='utf-8',style_compression=0)
    
    for task in range(args.num_works):
        print("Task {0}: ".format(task))
        with open(RESULT_PATH_VAL, 'a') as file_val:
            file_val.write('Task {0}: '.format(task) + '\n')
        
        start = time.time()

        if task != 0 and task < 21:
            model_ = copy.deepcopy(model)
            del model
            model = model_
            model.wider(task - 1)
            
        if args.cuda:
            model = model.cuda()
            
        print("Model Now...")
        model.printf()
        with open(RESULT_PATH_VAL, 'a') as file_val:
            file_val.write('Model Now...\n{}'.format(model) + '\n\n')
        print()
        
        args.lr = lr_
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,  weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_epoch = 0
        
        snapshot_comp = './val_outputs'
        snapshot = './val_outputs/Val_lifelong_scratch_cifar_imagenet/{0}/task_{1}'.format(now_time, task)
        #snapshot = './val_outputs/Val_lifelong_scratch_cifar_imagenet/2021-12-15 07:51:43/task_{0}'.format(task)
        
        if not os.path.isdir(snapshot):
            print("==> building model checkpoint file")
            os.makedirs(snapshot)
        checkpoint_path = os.path.join(snapshot, 'checkpoint.pth')
        
        if os.path.isfile(checkpoint_path):
            print('load the pretrained model parameters...')
            with open(RESULT_PATH_VAL, 'a') as file_val:
                file_val.write('load the pretrained model parameters...\n')
            
            checkpoint = torch.load(checkpoint_path)
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print('From Epoch {0}: ...'.format(best_epoch))
            
            with open(RESULT_PATH_VAL, 'a') as file_val:
                file_val.write('From Epoch {0}: ...\n\n'.format(best_epoch))
            
            print()
            
        loss_train = []
        acc_train = []
        train_name = 'train' + str(task)
        test_name = 'test' + str(task)
    
        if args.judgement:
            for name, module in modules:
                record_1_epoch[module] = 0.0
                record_10_epoch[module] = 0.0
                record_50_epoch[module] = 0.0
 
        
        sheet_task = book.add_sheet('Task_{0}'.format(task), cell_overwrite_ok=True)
        
        
        cnt_epoch = 0
        for epoch in range(best_epoch, args.epochs):
            
            modules = model.named_modules()
            
            print('Epoch {0}:'.format(epoch + 1))
            with open(RESULT_PATH_VAL, 'a') as file_val:
                file_val.write('Epoch {0}:\n'.format(epoch + 1))
                
            if args.judgement:
                for name, module in modules:
                    # print('Name: {0}, Module: {1}\n'.format(name, module))
                    record_1_epoch[module] = 0.0
                    if epoch % 10 == 0:
                        record_10_epoch[module] = 0.0
                    if epoch % 50 == 0:
                        record_50_epoch[module] = 0.0

                    
            start = time.time()
            is_best = False
                  
            # modules = model.named_modules()
            
            batch_num = 0
            
            train_loss, train_acc = train(train_loader[task](epoch), model, criterion, optimizer, epoch, args, snapshot=snapshot, name=train_name) #, RESULT_PATH_VAL=RESULT_PATH_VAL)
                
            batch_num = len(train_loader[task](epoch))
                
            print('Training: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
                train_loss,
                train_acc,
            ))
            with open(RESULT_PATH_VAL, 'a') as file_val:
                file_val.write('Training: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
                    train_loss,
                    train_acc,
                ))     
            
            loss_train.append(train_loss)
            acc_train.append(train_acc)

            save_checkpoint({
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                #'best_state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'train_acc': train_acc,
            }, is_best=is_best, output_dir=snapshot)
            
            # save best model
#             if test_acc > best_acc:
#                 is_best = True
#                 best_acc = test_acc
#                 save_checkpoint({
#                     'optimizer': optimizer.state_dict(),
#                     'state_dict': model.state_dict(),
#                     #'best_state_dict': model.module.state_dict(),
#                     'best_acc': best_acc,
#                     'epoch': epoch + 1,
#                     'train_acc': train_acc,
#                 }, is_best=is_best, output_dir=snapshot)
                
            finish = time.time()
            print('Epoch {} training time consumed: {:.2f}s'.format(epoch + 1, finish - start))
   
            
            with open(RESULT_PATH_VAL, 'a') as file_val:
                file_val.write('Gradient change statisitcs: Task {0}, Epoch {1}\n'.format(task, epoch + 1)) 
                
                sheet_task.write(cnt_epoch, 0, 'Epoch_{0}'.format(cnt_epoch))
                
                cnt = 1
                
                for modl, value in record_1_epoch.items():
                    if isinstance(modl, nn.Conv2d):
                        print(modl)
                        print(str(value) + '\n')
                        file_val.write('Module: {0}, ratio: {1}\n'.format(modl, 1.0 * value / batch_num))
                        
                        sheet_task.write(cnt_epoch, cnt, 1.0 * value / batch_num)
                        cnt = cnt + 1
                        
                file_val.write('\n\n')
            record_1_epoch.clear()
            cnt_epoch = cnt_epoch + 1
            
        #if (task + 1) % 10 == 0:
        #    book.save(r'./val_outputs/Val_lifelong_scratch_cifar_imagenet/{0}/record_change_{1}.xls'.format(now_time, task+1))
    
if __name__ == "__main__":
    main()

