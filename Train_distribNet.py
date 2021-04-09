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
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

import math
from Model_distributeNet import dis_net as disNet
from LossFunc_lossCalc import FeaturesLoss,UncertaintyLoss
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='tri_view', type=str, help='output model name')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir',default='./data/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batch_size', default=4, type=int, help='batchsize')
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
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
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

transform_train_list = [
        transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=transforms.InterpolationMode.BICUBIC), #Image.BICUBIC)
        transforms.Resize((opt.h, opt.w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        # 那transform.Normalize()是怎么工作的呢？
        # 以上面代码为例，ToTensor()能够把灰度范围从0-255变换到0-1之间，
        # 而后面的transform.Normalize()
        # 则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalize执行以下操作：

transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad( opt.pad, padding_mode='edge'),
        #transforms.RandomAffine(90),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),interpolation=transforms.InterpolationMode.BICUBIC), #Image.BICUBIC
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list) }


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'),
                                          data_transforms['satellite'])
image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'),
                                          data_transforms['train'])
image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'),
                                          data_transforms['train'])
image_datasets['google'] = datasets.ImageFolder(os.path.join(data_dir, 'google'),
                                          data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                             shuffle=False, num_workers=1, pin_memory=True) # 8 workers may work faster
              for x in ['satellite', 'street', 'drone', 'google']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'street', 'drone', 'google']}

class_names = image_datasets['street'].classes
print(dataset_sizes)
device = torch.device("cpu")
use_gpu = False
if use_gpu:
    device = torch.device("cuda")
    use_gpu = torch.cuda.is_available()

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
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            index=0;
            # Iterate over data.
            for data,data2,data3,data4 in zip(dataloaders['satellite'], dataloaders['street'], dataloaders['drone'], dataloaders['google']) :
                show_data.show(data,data2,data3,data4)
                # get the inputs
                feature_loss = 0
                unsertainty_loss = 0    
                if phase == 'train':
                    index+=1
                    if index % 2 == 1:
                        inputs, labels = data
                        inputs2, labels2 = data2
                        inputs3, labels3 = data3
                        inputs4, labels4 = data4
                        continue
                    else:
                        inputs5, labels5 = data
                        inputs6, labels6 = data2
                        inputs7, labels7 = data3
                        inputs8, labels8 = data4
                else:
                    inputs, labels = data
                    inputs2, labels2 = data2
                    inputs3, labels3 = data3
                    inputs4, labels4 = data4
                    
                now_batch_size,c,h,w = inputs.shape
                
                if now_batch_size<opt.batch_size: # skip the last batch
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    inputs2 = Variable(inputs2.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels2 = Variable(labels2.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                    
                    inputs5 = Variable(inputs5.cuda().detach())
                    inputs6 = Variable(inputs6.cuda().detach())
                    inputs7 = Variable(inputs7.cuda().detach())
                    
                    labels5 = Variable(labels5.cuda().detach())
                    labels6 = Variable(labels6.cuda().detach())
                    labels7 = Variable(labels7.cuda().detach())
                    
                    if opt.extra_Google:
                        inputs4 = Variable(inputs4.cuda().detach())
                        labels4 = Variable(labels4.cuda().detach())
                        inputs8 = Variable(inputs8.cuda().detach())
                        labels8 = Variable(labels8.cuda().detach())
                #else:
                #    inputs, labels = Variable(inputs), Variable(labels)
                #    inputs5, labels5 = Variable(inputs5), Variable(labels5)
 
 
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                if phase == 'search':
                    with torch.no_grad():
                        result = model(inputs2)
                        #just for ground
                elif phase == 'cal_database':
                    with torch.no_grad():
                        result = model(inputs, inputs3)
                        #just for satelite and drone 
                else:
                    if opt.views == 2: 
                        result = model(inputs, inputs2, inputs5, inputs6)
                         #(1,2,5,6)
                        #just for satelite with ground
                    elif opt.views == 3: 
                        if opt.extra_Google:
                            result = model(inputs, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8)
                            #(1,2,3,4,5,6,7,8)
                        else:
                            result = model(inputs, inputs2, inputs3, inputs5, inputs6, inputs7)
                            #(1,2,3,5,6,7)
                if opt.views == 2: 
                    0
                if opt.views == 3:
                    for i in range(3):
                        for j in range(i+1,3):
                            anchor = result[i]
                            positive = result[j]
                            negative = result[j+3]
                            #out = (长度为numclasss的均值向量，方差向量，【N*samples向量】) 
                            feature_loss1 = FeaturesLoss(
                                manchor=anchor[0],sanchor=anchor[2],
                                mpositive=positive[0],spositvie=positive[2],
                                mnegative=negative[0],K=opt.loss_k)
                            
                            unsertainty_loss1 = UncertaintyLoss(disanchor=anchor[1])
                            
                            anchor = result[i+3]
                            positive = result[j+3]
                            negative = result[j]
                            feature_loss2 = FeaturesLoss(
                                manchor=anchor[0],sanchor=anchor[2],
                                mpositive=positive[0],spositvie=positive[2],
                                mnegative=negative[0],K=opt.loss_k)
                            
                            unsertainty_loss2 = UncertaintyLoss(disanchor=anchor[1])
                            
                            feature_loss += feature_loss1 + feature_loss2
                            unsertainty_loss += unsertainty_loss1 + unsertainty_loss2            
                            
                if phase == 'train':
                    #if fp16: # we use optimier to backward loss
                    #    with amp.scale_loss(feature_loss, optimizer),amp.scale_loss( unsertainty_loss , optimizer) as scaled_floss,scaled_dloss :
                    #        scaled_floss,scaled_dloss
                    #        scaled_floss.backward(retain_graph=True)
                    #        if scaled_dloss < opt.loss_lamda * 6.0:
                    #            scaled_dloss = opt.loss_lamda * 6.0 - unsertainty_loss;
                    #            scaled_dloss.backward();
                    #else:
                    total_loss = feature_loss
                    if unsertainty_loss < opt.loss_lamda :
                        runsertainty_loss = opt.loss_lamda - unsertainty_loss;
                        total_loss += runsertainty_loss
                    total_loss.backward()
                if opt.views == 3:
                  print('[epoch:%d, iter:%d/%d] feature_loss: %.05f | real_unsertainty-Loss: %.05f ' 
                        % (epoch + 1, (index + 1 + epoch *  dataset_sizes['satellite']),  dataset_sizes['satellite']/opt.batch_size ,feature_loss,unsertainty_loss ))

            if phase == 'train':
                scheduler.step()
                
            #last_model_wts = model.state_dict()
            #if epoch%20 == 19:
            save_network(model, opt.name, epoch)
            #draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    #save_network(model_test, opt.name+'adapt', epoch)

    return model


######################################################################
# Draw Curve
#---------------------------
#x_epoch = []
#fig = plt.figure()
#ax0 = fig.add_subplot(121, title="loss")
#ax1 = fig.add_subplot(122, title="top1err")
#def draw_curve(current_epoch):
#    x_epoch.append(current_epoch)
#    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
#    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
#    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
#    if current_epoch == 0:
#        ax0.legend()
#        ax1.legend()
#    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#


#resnet = torch.load('three_view')
model = disNet(ccuda = use_gpu).to(device)
#print(model)
# For resume:
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

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
#dir_name = os.path.join('./model',name)
#if not opt.resume:
#    if not os.path.isdir(dir_name):
#        os.mkdir(dir_name)
#record every run
#    copyfile('./Train_distribNet.py', dir_name+'/train.py')
#    copyfile('./Model_distributeNet.py', dir_name+'/model.py')
# save opts
#    with open('%s/opts.yaml'%dir_name,'w') as fp:
#        yaml.dump(vars(opt), fp, default_flow_style=False)
# model to gpu

#if fp16:
#    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")
if __name__ == '__main__':
    model = train_model(model, FeaturesLoss, UncertaintyLoss, optimizer_ft, exp_lr_scheduler)

