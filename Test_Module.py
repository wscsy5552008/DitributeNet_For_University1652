# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:25:04 2021

@author: Jinda
"""

from __future__ import print_function, division
import argparse
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import show_data
#import matplotlib
#matplotlib.use('agg')
#from PIL import Image
import time
from Data_presolveing import getdatasets
import os
from random_erasing import RandomErasing
from utils import load_network, save_network
version =  torch.__version__
from torch.utils.data import DataLoader
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

import math
from Model_resNet import PreTrainResNet as resNet
from LossFunc_lossCalc import FeaturesLoss,UncertaintyLoss,FeaturesLossWithoutSample

def cal_database(model,loader):
    for index,items in enumerate(loader, 0) :
        #items = (label,pic)
        label,pic = items
        result = model(pic)
        
        #just for satllite
def cal_resNet_sate(inputs):
        return resNet(x2 = inputs)

def test_model(model, FeaturesLoss, UncertaintyLoss, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    #warm_up = 0.1 # We start from the 0.1*lrRate
    #warm_iteration = round(dataset_sizes['satellite']/opt.batch_size)*opt.warm_epoch # first 5 epoch

    for epoch in range(num_epochs-start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase

        model.train(True)  # Set model to training mode

        index=0;
        # Iterate over data.
        for index,items in enumerate(train_loader,0) :
            g1,d1,s1,g2,d2,s2 = items
            #show_data.show(g1,d1,s1,g1,'%dt1.jpg'%index)
            #show_data.show(g2,d2,s2,g2,'%dt2.jpg'%index)
            if use_gpu:
                g1,d1,s1 = Variable(g1.cuda().detach()),Variable(d1.cuda().detach()),Variable(s1.cuda().detach())
                g2,d2,s2 = Variable(g2.cuda().detach()),Variable(d2.cuda().detach()),Variable(s2.cuda().detach())
               
            # zero the parameter gradients
            optimizer.zero_grad()
            
           
            result = model(g1,d1,s1,g2,d2,s2)
                  
            feature_loss = 0.0
            
            for i in range(3):
                for j in range(i+1,3):
                    anchor = result[i]
                    positive = result[j]
                    negative = result[j+3]
                    #out = (长度为numclasss的均值向量，方差向量，【N*samples向量】) 
                    feature_loss1 = FeaturesLossWithoutSample(
                        manchor=anchor,
                        mpositive=positive,
                        mnegative=negative)
                    
                    anchor = result[i+3]
                    positive = result[j+3]
                    negative = result[j]
                    feature_loss2 = FeaturesLossWithoutSample(
                        manchor=anchor,
                        mpositive=positive,
                        mnegative=negative)
                    
                    feature_loss += feature_loss1 + feature_loss2
            
            optimizer.zero_grad()
            feature_loss.backward()
            optimizer.step()
            print('[epoch:%d, iter:%d/%d] feature_loss: %.05f' 
                  % (epoch + 1, index, len(train_loader) ,feature_loss ))

        save_network(model, opt.name, epoch)
       
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()
    return model

model = resNet().to(device)