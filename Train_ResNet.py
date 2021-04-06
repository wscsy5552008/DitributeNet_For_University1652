# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:17:31 2021

@author: Jinda
"""
# -*- coding: utf-8 -*-

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
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='resnet34', type=str, help='output model name')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir',default='./data/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batch_size', default=3, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--views', default=3, type=int, help='the number of views')
parser.add_argument('--loss_lamda', default=16, type=int, help='Ditribute_Loss Lamda' )
parser.add_argument('--loss_k',  default=1, type=int, help='frature_loss K' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
parser.add_argument('--share', action='store_true', help='share weight between different view' )
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google' )
parser.add_argument('--fp16', action='store_false', help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0


fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#


    
#trainImgSet = torch.load('dataTemp/Datasets100')#getdatasets()
 
trainImgSet = getdatasets()

train_loader = DataLoader(dataset=trainImgSet,batch_size=opt.batch_size ,shuffle=False)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
device = torch.device("cpu")
use_gpu = False
if use_gpu:
    device = torch.device("cuda")
    use_gpu = torch.cuda.is_available()
    
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, FeaturesLoss, UncertaintyLoss, optimizer, scheduler, num_epochs=25):
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
            gr1,dr1,sr1,gr2,dr2,sr2 = result
            
            #1 ground and satellite
            #feature_loss_gs1 =  FeaturesLossWithoutSample(
            #            manchor=gr1,
            #            mpositive=sr1,
            #            mnegative=sr2)
            
            #1 drone and satellite
            feature_loss_ds1 =  FeaturesLossWithoutSample(
                        manchor=dr1,
                        mpositive=sr1,
                        mnegative=sr2)
            #2 ground and satellite
            #feature_loss_gs2 =  FeaturesLossWithoutSample(
            #            manchor=gr2,
            #            mpositive=sr2,
            #            mnegative=sr1)
            
            #2 drone and satellite
            feature_loss_ds2 =  FeaturesLossWithoutSample(
                        manchor=dr2,
                        mpositive=sr2,
                        mnegative=sr1)
            
            #feature_loss = feature_loss_gs1 + feature_loss_gs2 + 
            feature_loss = feature_loss_ds1 + feature_loss_ds2
                
            #for i in range(2):
            #    for j in range(i+1,3):
            #        anchor = result[i]
            #        positive = result[j]
            #        negative = result[j+3]
                    #out = (长度为numclasss的均值向量，方差向量，【N*samples向量】) 
            #        feature_loss1 = FeaturesLossWithoutSample(
            #            manchor=anchor,
            #            mpositive=positive,
            #            mnegative=negative)
                    
            #        anchor = result[i+3]
            #        positive = result[j+3]
            #        negative = result[j]
            #        feature_loss2 = FeaturesLossWithoutSample(
            #            manchor=anchor,
            #            mpositive=positive,
            #            mnegative=negative)
                    
             #       feature_loss += feature_loss1 + feature_loss2
            
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

if start_epoch>=40:
    opt.lr = opt.lr*0.1

ignored_params = list(map(id, model.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)

if __name__ == '__main__':
    model = train_model(model, FeaturesLoss, UncertaintyLoss, optimizer_ft, exp_lr_scheduler)
