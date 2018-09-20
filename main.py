#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:30:08 2017

@author: wen37
"""
from train import train_prednet
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--model', default='PredNetLocal',  help='model used')
parser.add_argument('--cls', default=3, type=int, help='cycles')
parser.add_argument('--gpu', default=4, type=int, help='number of gpu')
parser.add_argument('--lr', default=0.01,type=float, help='learning rate')
args = parser.parse_args()

if __name__ == '__main__':
    train_prednet(cnn=False, model=args.model, circles=args.cls, gpunum=args.gpu,lr=args.lr)

