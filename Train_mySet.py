# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:17:31 2021

@author: Jinda
"""
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse

import os
import yaml
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

#MODELPATH = "C:\\Users\\Jinda\\Desktop\\源代码\\university1652-model\\three_view_long_share_d0.75_256_s1_google\\net_119.pth"
import Par_train as para
from Data_presolveing import getdatasets
from utils import load_network, save_network
from Model_distributeNet import three_view_net,three_view_resNet,two_view_net
from LossFunc_lossCalc import FeaturesLoss,UncertaintyLoss,FeaturesLossWithoutSample

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
#################################################################
#model 存储的名字
parser.add_argument('--name',default='three_view', type=str, help='output model name')
#路径
parser.add_argument('--data_dir',default='../',type=str, help='training dir path')
#一些设置
parser.add_argument('--loss_lamda', default=8, help='uncertainty lamda' )
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google' )
parser.add_argument('--views', default=1, type=int, help='the number of views')
parser.add_argument('--share', action='store_true', help='share weight between different view' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--batch_size', default=2, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--pool',default='avg', type=str, help='pool avg')
#似乎没有用到，是否训练所有数据集
parser.add_argument('--train_all', action='store_true', help='use all training data' )
#gpu_id 
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
#加载模型基础上继续训练
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
#宽高不用说了
parser.add_argument('--h', default=para.H, type=int, help='height')
parser.add_argument('--w', default=para.W, type=int, help='width')
#droprate
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
#学习率不用说了
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
#################################################################

parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )

parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
#每个epoch模型按照比例与之前的模型结合，avg表示原模型的比率，如果为 >= 1，失效

parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
#学习率先小后大   有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳有助于保持模型深层的稳定性
#设置为0则说明不使用，否则就是每 每隔几代，warm_up增加直至到1
#    warm_up = 0.1 # We start from the 0.1*lrRate
#    warm_iteration = round(dataset_sizes['satellite']/opt.batchsize)*opt.warm_epoch # first 5 epoch
# if epoch < opt.warm_epoch and phase == 'train': 
#                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
#                    loss *= warm_up
#################################################################
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
#单独图片的数据增强，亮度啊之类的会进行随机更改，但是并不增加图片数量
#if opt.DA:
#    transform_train_list = [ImageNetPolicy()] + transform_train_list  / transform is a img transform

parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
#color的一个抖动，效果应该和上述类似
#if opt.color_jitter:
#    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
#    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--start', default=0, type=int, help='the first K epoch that needs warm up')
#随机擦除一些数据点，效果和上述类似 Randomly selects a rectangle region in an image and erases its pixels.
#if opt.erasing_p>0:
#    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]
#################################################################
opt = parser.parse_args()

MODELPATH = 'model\\three_view\\net_074.pth'

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = opt.start


fp16 = opt.fp16
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
use_gpu =para.USE_GPU
if use_gpu:
    device = torch.device("cuda")
    use_gpu = torch.cuda.is_available()
    
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
groundFeaturePath = "train_log/groundFeature.txt"
gr = open(groundFeaturePath,'w')

def train_model(model, optimizer, scheduler, num_epochs=50):
    
    since = time.time()
    warm_up = 0.1 # We start from the 0.1*lrRate
    model.train(True)  # Set model to training mod
    model.to(device)
    
    # zero the parameter gradients
    for epoch in range(start_epoch,num_epochs):
        ######################################################################
        # Load Data
        # ---------
        #
        trainImgSet = None
        trainImgSet = getdatasets(opt)
        if len(trainImgSet) < para.MINI:
            trainImgSet = getdatasets(opt)

        train_loader = DataLoader(dataset=trainImgSet,batch_size=opt.batch_size ,shuffle=False)
        del(trainImgSet)

        totalDS = 0.0
        totalGD = 0.0
        totalGS = 0.0
        if para.IS_DIS_Net:
            filepath = 'train_log/dis_train_log'+str(epoch)+'.txt'
        else:
            filepath = 'train_log/res_train_log'+str(epoch)+'.txt'

        logfile = open(filepath, "w")   
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('-' * 10,file=logfile)
        # Each epoch has a training and validation phase
        # Iterate over data.
        
        for index,items in enumerate(train_loader,0) : 
            now_batch_size = len(items[0])
            if now_batch_size<opt.batch_size: # skip the last batch
                continue
            g1,d1,s1,g2,d2,s2 = items
            #show_data.show(g1,d1,s1,g1,'%dt1.jpg'%index)
            #show_data.show(g2,d2,s2,g2,'%dt2.jpg'%index)
            if use_gpu:
                g1,d1,s1 = Variable(g1.cuda().detach()),Variable(d1.cuda().detach()),Variable(s1.cuda().detach())
                g2,d2,s2 = Variable(g2.cuda().detach()),Variable(d2.cuda().detach()),Variable(s2.cuda().detach())
              
            if opt.views == 1:
                result1 = model(satellite = s1, ground = g1, drone = None)
                result2 = model(satellite = s2, ground = g2, drone = None)
            elif opt.views == 2:
                result1 = model(satellite = None, ground = g1, drone = d1)
                result2 = model(satellite = None, ground = g2, drone = d2)
            elif opt.views == 3:
                result1 = model(satellite = s1, ground = g1, drone = d1)
                result2 = model(satellite = s2, ground = g2, drone = d2)
            elif opt.views == 4:
                result1 = model(satellite = s1, ground = None, drone = d1)
                result2 = model(satellite = s2, ground = None, drone = d2)
                  
                  
            feature_loss = torch.zeros(size=(1,1),dtype = float)
            uncertainty_loss =  torch.zeros(size=(1,1),dtype = float) 
            
            if use_gpu:
                feature_loss =  Variable(feature_loss.cuda().detach())
                uncertainty_loss =  Variable(uncertainty_loss.cuda().detach())
                
            
            print('[epoch:%d, iter:%d/%d]' 
                  % (epoch + 1, index, len(train_loader)) ,file=gr)
            print(result1,file=gr)
            
            if para.IS_DIS_Net:
                sr1,gr1,dr1 = result1
                sr2,gr2,dr2 = result2
                
                if gr1 == None :
                    #1 drone and satellite
                    feature_loss_ds1 =  FeaturesLoss(
                                manchor=dr1[0],sanchor=dr1[2],
                                mpositive=sr1[0],spositvie=sr1[2],
                                mnegative=sr2[0],snegative=sr2[2])
                        
                    #2 drone and satellite
                    feature_loss_ds2 =  FeaturesLoss(
                                manchor=dr2[0],sanchor=dr2[2],
                                mpositive=sr2[0],spositvie=sr2[2],
                                mnegative=sr1[0],snegative=sr1[2])

                    DSLoss = feature_loss_ds1 + feature_loss_ds2
                    feature_loss = feature_loss + DSLoss
                    totalDS+=DSLoss
                if sr1 == None :
                    #1 drone and ground
                    feature_loss_dg1 =  FeaturesLoss(
                                manchor=gr1[0],sanchor=gr1[2],
                                mpositive=dr1[0],spositvie=dr1[2],
                                mnegative=dr2[0],snegative=dr2[2])
                        
                    #2 drone and ground
                    feature_loss_dg2 =  FeaturesLoss(
                                manchor=gr2[0],sanchor=gr2[2],
                                mpositive=dr2[0],spositvie=dr2[2],
                                mnegative=dr1[0],snegative=dr1[2])

                    GDLoss = feature_loss_dg1 + feature_loss_dg2
                    feature_loss = feature_loss + GDLoss
                    totalGD+=GDLoss
                
                if gr1 !=None and sr1 != None:
                    #1 ground and satellite. drone detach?No
                
                    feature_loss_gs1 =  FeaturesLoss(
                                manchor=gr1[0],sanchor=gr1[2],
                                mpositive=sr1[0],spositvie=sr1[0],
                            mnegative=sr2[0],snegative=sr2[2])
                   
                    #2 ground and satellite
                   
                    feature_loss_gs2 =  FeaturesLoss(
                            manchor=gr2[0],sanchor=gr2[2],
                            mpositive=sr2[0],spositvie=sr2[0],
                            mnegative=sr1[0],snegative=sr1[2])
    
                    GSLoss = feature_loss_gs1 + feature_loss_gs2
                    feature_loss = feature_loss + GSLoss
                    totalGS+=GSLoss

                for item in result1:
                    if item != None:
                        uncertainty_loss = uncertainty_loss +  UncertaintyLoss(disanchor=item[1])
                for item in result2:
                    if item != None:
                        uncertainty_loss = uncertainty_loss +  UncertaintyLoss(disanchor=item[1])

                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch : 
                    warm_up = min(1.0, warm_up + 0.9 / opt.warm_epoch)
                    feature_loss *= warm_up
                    uncertainty_loss *= warm_up

                if uncertainty_loss < opt.loss_lamda :
                    runcertainty_loss = opt.loss_lamda - uncertainty_loss
                    feature_loss = feature_loss + runcertainty_loss
            else :
                
                sr1,gr1,dr1 = result1
                sr2,gr2,dr2 = result2
                if gr1 == None :
                
                    #1 drone and satellite
                    feature_loss_ds1 =  FeaturesLossWithoutSample(
                            manchor=dr1,
                            mpositive=sr1,
                            mnegative=sr2)
                    #2 drone and satellite
                    feature_loss_ds2 =  FeaturesLossWithoutSample(
                                manchor=dr2,
                                mpositive=sr2,
                                mnegative=sr1)

                    DSLoss = feature_loss_ds2 + feature_loss_ds2
                    feature_loss = feature_loss + DSLoss
                    totalDS+=DSLoss
                
                if sr1 == None :
                    feature_loss_gd1 =  FeaturesLossWithoutSample(
                                manchor=gr1,
                                mpositive=dr1,
                                mnegative=dr2)
                    feature_loss_gd2 =  FeaturesLossWithoutSample(
                                manchor=gr2,
                                mpositive=dr2,
                                mnegative=dr1)
                    GDLoss = feature_loss_gd1 + feature_loss_gd2
                    feature_loss = feature_loss + GDLoss
                    totalGD+=GDLoss

                if gr1 !=None and sr1 != None:
                    #1 ground and satellite. drone detach?No
                    
                    feature_loss_gs1 =  FeaturesLossWithoutSample(
                                manchor=gr1,
                                mpositive=sr1,
                                mnegative=sr2)
                
                    #2 ground and satellite
                    feature_loss_gs2 =  FeaturesLossWithoutSample(
                                manchor=gr2,
                                mpositive=sr2,
                                mnegative=sr1)
                    GSLoss = feature_loss_gs1 + feature_loss_gs2
                    feature_loss = feature_loss + GSLoss
                    totalGS+=GSLoss

                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch : 
                    warm_up = min(1.0, warm_up + 0.9 / opt.warm_epoch)
                    feature_loss *= warm_up

            if fp16: # we use optimier to backward loss
                with amp.scale_loss(feature_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                feature_loss.backward()
                
            optimizer.step()
            optimizer.zero_grad()
            
            if para.IS_DIS_Net:
                print('[epoch:%d, iter:%d/%d] feature_loss : %8.05f | unsertainty-Loss: %.05f ' 
                          % (epoch , index, len(train_loader) , feature_loss,uncertainty_loss ))
                print('[epoch:%d, iter:%d/%d] feature_loss : %8.05f | unsertainty-Loss: %.05f ' 
                          % (epoch , index, len(train_loader) ,feature_loss,uncertainty_loss ),file=logfile)
            else:
                print('[epoch:%d, iter:%d/%d] feature_loss : %8.05f ' 
                          % (epoch , index, len(train_loader) , feature_loss))
                print('[epoch:%d, iter:%d/%d] feature_loss : %8.05f ' 
                          % (epoch , index, len(train_loader) , feature_loss),file=logfile)

        logfile.close()
        scheduler.step()
        time_elapsed = time.time() - since
        save_network(model, opt.name, epoch + 1)
        print('Training complete in .%0fm :%.0fs : avgDroneSatLoss:%.05f | avgGroDroneLoss:%.05f | avgGroSatelliteLoss:%.05f'%(
            time_elapsed // 60, time_elapsed % 60, totalDS/len(train_loader), totalGD/len(train_loader), totalGS/len(train_loader)))
        print()


    time_elapsed = time.time() - since
    print('Training complete in  in .%0fm :%.0fs'%(
        time_elapsed // 60, time_elapsed % 60))
    print()
    return model

######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',opt.name,'train.jpg'))
######################################################################


if __name__ == '__main__':
    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #

    #model = three_view_net(use_gpu=use_gpu).to(device)
    if para.IS_DIS_Net:
        if opt.views == 3:
            model = three_view_net(use_gpu=use_gpu,resnetNum=18).to(device)
        else :   
            model = two_view_net(use_gpu=use_gpu,resnetNum=18).to(device)

    else :
        model = three_view_resNet(resnetNum = 34).to(device)
        opt.lr*=10
    model.load_state_dict(torch.load(MODELPATH))

    # For resume:
    if start_epoch>=40:
        opt.lr = opt.lr*0.1
    if para.IS_DIS_Net:
        ignored_params = list(map(id, model.disblock_1.parameters() )) + list(map(id, model.disblock_2.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                    {'params': base_params, 'lr': 0.1*opt.lr},
                    {'params': model.disblock_1.parameters(), 'lr': opt.lr},
                    {'params': model.disblock_2.parameters(), 'lr': opt.lr}
                ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        optimizer_ft = optim.SGD(params=model.parameters(),lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        
    #ignored_params = list(map(id, model.classifier.parameters() )) #返回一串parameters的标识符， 因为classifier层是被忽略的，所以这里是ignore
    #base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())#过滤掉这些元素，这其实可以直接拿来用，因为我也不需要这里的classsifier
    #optimizer_ft = optim.SGD([
    #         {'params': base_params, 'lr': 0.1*opt.lr},
    #         {'params': model.classifier.parameters(), 'lr': opt.lr}
    #     ], weight_decay=5e-4, momentum=0.9, nesterov=True)
#这里的意思是设置不同的学习率
    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)


    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    #

    dir_name = os.path.join('./model',opt.name)

    if not opt.resume:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
    # save opts
        with open('%s/opts.yaml'%dir_name,'w') as fp:
            yaml.dump(vars(opt), fp, default_flow_style=False)

    if fp16:
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")
        print("fp")
        
    model = train_model(model,  optimizer_ft, exp_lr_scheduler,num_epochs=para.EPOCH)

    save_network(model, opt.name, 200)