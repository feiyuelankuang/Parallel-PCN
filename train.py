'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from models import *
from utils import progress_bar
from torch.autograd import Variable

def train_prednet(lr = 0.01,cnn=False, model='PredNetTied', circles=2, gpunum=2):
    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batchsize = 256

    root = './'
    lr = lr
    
    models ={'PredictiveTied_LN_Multi':PredictiveTied_LN_Multi,'PredictiveTied_GN': PredictiveTied_GN,'PredictiveTiedKL':PredictiveTiedKL,'PredictiveTiedInstanceNorm':PredictiveTiedInstanceNorm,'PredictiveTiedLayerNorm':PredictiveTiedLayerNorm,'PredictiveTied3':PredictiveTied3,'PredictiveTied':PredictiveTied,'PredictiveComb':PredictiveComb,'PredictiveSELU':PredictiveSELU,'PredictiveNew':PredictiveNew,'Predictive': Predictive,'PredictiveFull': PredictiveFull, 'PredNetNew': PredNetNew, 'PredNet': PredNet, 'PredNetLocal': PredNetLocal,'PredictiveNew2':PredictiveNew2,'PredictiveNew3':PredictiveNew3,'PredictiveTied_LN':PredictiveTied_LN}
    if cnn:
        modelname = model+'_'
    else:
        modelname = model+'_'+str(circles)+'CLS_'
    
    # clearn folder
    checkpointpath = root+'checkpoint/'
    logpath = root+'log/'
    if not os.path.isdir(checkpointpath):
        os.mkdir(checkpointpath)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
        
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    
    # Model
    print('==> Building model..')
    if cnn:
        net = models[model](num_classes=100)
    else:
        net = models[model](num_classes=100,cls=circles)
       
#    print(1)
    # Define objective function
    criterion = nn.CrossEntropyLoss()
 #   print(2)
    if model == 'PredictiveComb':
        convparas = [p for p in net.FFconv.parameters()]+\
                        [p for p in net.FBconv.parameters()]+\
                        [p for p in net.linear.parameters()]+\
                        [p for p in net.comb.parameters()]
        optimizer = optim.SGD([{'params': convparas},
                    ], lr=lr, momentum=0.9, weight_decay=5e-4)
    elif model == 'PredictiveSELU':
        convparas = [p for p in net.FFconv.parameters()]+\
                        [p for p in net.FBconv.parameters()]+\
                        [p for p in net.linear.parameters()]+\
                        [p for p in net.selu.parameters()]
        rateparas = [p for p in net.b0.parameters()]
        optimizer = optim.SGD([
                    {'params': convparas},
                    {'params': rateparas, 'weight_decay': 0},
                    ], lr=lr, momentum=0.9, weight_decay=5e-4)


    elif model == 'PredictiveTied3':
        convparas = [p for p in net.conv.parameters()]+\
                        [p for p in net.linear.parameters()]
        rateparas = [p for p in net.a0.parameters()]+\
                    [p for p in net.b0.parameters()]+\
                    [p for p in net.c0.parameters()]
        optimizer = optim.SGD([
                    {'params': convparas},
                    {'params': rateparas, 'weight_decay': 0},
                    ], lr=lr, momentum=0.9, weight_decay=5e-4)

    elif model == 'PredictiveTiedKL':
        convparas = [p for p in net.conv.parameters()]+\
                        [p for p in net.linear.parameters()]
        rateparas = [p for p in net.b0.parameters()]

        optimizer = optim.SGD([
                    {'params': convparas},
                    {'params': rateparas, 'weight_decay': 0},
                    ], lr=lr, momentum=0.9, weight_decay=5e-4)

    elif 'PredictiveTied' in model:
        convparas = [p for p in net.conv.parameters()]+\
                        [p for p in net.linear.parameters()]
        rateparas = [p for p in net.a0.parameters()]+\
                    [p for p in net.b0.parameters()]
        optimizer = optim.SGD([
                    {'params': convparas},
                    {'params': rateparas, 'weight_decay': 0},
                    ], lr=lr, momentum=0.9, weight_decay=5e-4)

    else:
        convparas = [p for p in net.FFconv.parameters()]+\
                        [p for p in net.FBconv.parameters()]+\
                        [p for p in net.linear.parameters()]
        rateparas = [p for p in net.a0.parameters()]+\
                    [p for p in net.b0.parameters()]
        optimizer = optim.SGD([
                    {'params': convparas},
                    {'params': rateparas, 'weight_decay': 0},
                    ], lr=lr, momentum=0.9, weight_decay=5e-4)

    print(3)

    # Parallel computing
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(gpunum))
#        net = torch.nn.DataParallel(net, device_ids=[3])
        cudnn.benchmark = True


    
    print(4)
   # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0          
        training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (batchsize, epoch, optimizer.param_groups[0]['lr'])
        statfile.write('\nTraining Setting: '+training_setting+'\n')
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            #print(outputs.size())
            #print(targets.size())
            if model == 'PredictiveTied_LN_Multi':
                loss = 0
                for i in range (len(outputs)):
                    loss = loss+ criterion(outputs[i], targets)
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
#            train_loss += loss.data[0]
            train_loss += loss.item()
            if model == 'PredictiveTied_LN_Multi':
                _, predicted = torch.max(outputs[-1].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
  
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total))
        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
                  % (epoch, train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)
        statfile.write(statstr+'\n')
        return train_loss/(batch_idx+1)
         
    # Testing
    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
#            print(outputs.size(),targets.size())
            if model == 'PredictiveTied_LN_Multi':
                loss = 0
                for i in range (len(outputs)):
                    loss = criterion(outputs[i], targets)
                    test_loss += loss.item()
            else:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            if model == 'PredictiveTied_LN_Multi':
                _, predicted = torch.max(outputs[-1].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1),100.*(float)(correct)/(float)(total), correct, total))
        statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
                  % (epoch, test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)
        statfile.write(statstr+'\n')
        
        # Save checkpoint.
        acc = 100.*correct/total
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
 #           'net': net.module if use_cuda else net,
        }
        torch.save(state, checkpointpath + modelname + '_last_ckpt.t7')
        if acc >= best_acc:
            print('Saving..')
            torch.save(state, checkpointpath + modelname + '_best_ckpt.t7')
            best_acc = acc
        
    # Set adaptive learning rates
    def decrease_learning_rate():
        """Decay the previous learning rate by 2"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2
        print('decreasing lr')
    def decrease_learning_rate_large():
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
        print('decreasing lr')
    
    LossList= []
    for epoch in range(start_epoch, start_epoch+200):
        statfile = open(logpath+'training_stats_'+modelname+'_with'+'.txt', 'a+')
        if len(LossList) < 5:
            LossList.append(train(epoch))
        else:
            LossList[0] = LossList[1]
            LossList[1] = LossList[2]
            LossList[2] = LossList[3]
            LossList[3] = LossList[4]
            LossList[4] = train(epoch)
            slope = (-2*(float)(LossList[0])-(float)(LossList[1])+(float)(LossList[3])+2*(float)(LossList[4]))/(float)(LossList[4])
            print(-slope)
            if model=='PridictiveRELU':
                if -slope < 0.5:
                    decrease_learning_rate()
                elif LossList[4] < 0.005:
                    break

            elif epoch % 10 == 0:
                if -slope > 0.3 and -slope < 0.5:
                    decrease_learning_rate()
                elif -slope < 0.3:
                    decrease_learning_rate_large()
                if LossList[4] < 0.01:
                    break
                   
        test(epoch)
