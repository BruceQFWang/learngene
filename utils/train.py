import time
import sys
sys.path.append('../')
import copy
import numpy as np
import os
import shutil
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from .Ewc_class import EWC, EWC_vgg

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    
def train(train_loader, model, criterion, optimizer, epoch, args, snapshot, name):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    
    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda :
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)
            
    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Training Epoch: '+str(epoch)+' Test set: Average loss: '+str(losses.sum / len(train_loader.dataset))+ \
           ', Accuracy: '+ str(top1.avg) +'\n')

    return losses.avg, top1.avg


def test(test_loader, model, criterion, optimizer, epoch, args, snapshot, name):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(len(test_loader), [batch_time, losses, acc], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        
        for i, (images, target) in enumerate(test_loader):
            if args.cuda :
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = model(images)
            
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            acc.update(acc1[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Test-Acc {top1.avg:.3f} '.format(top1=acc))
        
    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(' Test set: Average loss: '+str(losses.sum / len(test_loader.dataset))+', Accuracy: '+str(acc.avg)+'\n')

    return losses.sum/(len(test_loader.dataset)), acc.sum/(len(test_loader.dataset))


def train_ewc(train_loader, model, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))
    
    ewc = EWC(model, args.cuda)

    model.train()

    end = time.time()
    
    for i, (images, targets) in enumerate(train_loader):
        
        data_time.update(time.time() - end)

        if args.cuda :
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)
        
        ewc_loss = ewc.ewc_loss(args.cuda)
        loss = ce_loss + ewc_loss*1000

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)
    
    #os.system("nvidia-smi")
    #train_sample_loader = old_task
    
    ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print(' Done!')

    
    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Training Epoch: '+str(epoch)+' Test set: Average loss: '+str(losses.sum / len(train_loader.dataset))+
            ', Accuracy: '+str(top1.avg)+'\n')

    return losses.avg, top1.avg


def test_ewc(test_loader, model, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))
    
    ewc = EWC(model, args.cuda)
    
    model.eval()

    end = time.time()
    
    with torch.no_grad():
        
        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda :
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            ewc_loss = ewc.ewc_loss(args.cuda)
            loss = ce_loss + ewc_loss*1000
            
            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
            
        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task
    
    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')
    
    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: '+str(epoch)+' Test set: Average loss: '+str(losses.sum / len(test_loader.dataset))+', Accuracy: '+str(top1.avg)+'\n')

    return losses.avg, top1.avg


def train_ewc_vgg(train_loader, model, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    
    # print(len(train_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    ewc = EWC_vgg(model, args.cuda)

    model.train()

    end = time.time()
    
    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda :
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)
        
        ewc_loss = ewc.ewc_loss(args.cuda)
        loss = ce_loss + ewc_loss*1000
        #loss = criterion(outputs, targets)


        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    #os.system("nvidia-smi")
    #train_sample_loader = old_task
    
    ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Training Epoch: '+str(epoch)+' Test set: Average loss: '+str(losses.sum / len(train_loader.dataset))+
            ', Accuracy: '+str(top1.avg)+'\n')

    return losses.avg, top1.avg


def test_ewc_vgg(test_loader, model, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))
    
    ewc = EWC_vgg(model, args.cuda)

    model.eval()

    end = time.time()
    
    with torch.no_grad():
        
        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda :
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            ewc_loss = ewc.ewc_loss(args.cuda)
            loss = ce_loss + ewc_loss*1000
            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
        
        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task
    
    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')
    
    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: '+str(epoch)+' Test set: Average loss: ' + str(losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def sig(test_loader, model_one, args, snapshot, name='test', fisher_estimation_sample_size=128):
    print(len(test_loader.dataset))
   
    RESULT_PATH = os.path.join(snapshot, name)
    #RESULT_PATH = './outputs/VGG_lifelong_classes_five/vgg_comp_acc-{}'.format(name)
    file = open(RESULT_PATH, 'a')
    
    with torch.no_grad():
        cout = 0 
        for i, (images, targets) in enumerate(test_loader):
            # measure data loading time
            
            if args.cuda :
                images = images.cuda(non_blocking=True)
                #targets = targets.cuda(non_blocking=True)
            targets = targets.tolist()
            
            # compute output
            outputs_one = model_one(images).tolist()
            
            print(len(outputs_one[1]))
            for i in range(len(targets)):
                cout = cout + 1
                sigma_one = outputs_one[i][targets[i]]
                file.write('count: '+str(cout)+' sigma_one: '+str(float(sigma_one))+'\n')

    return sigma_one
