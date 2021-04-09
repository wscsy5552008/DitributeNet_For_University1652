# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:37:14 2021

@author: Jinda
"""

from numpy.lib.shape_base import _replace_zero_by_x_arrays
import torch
import torch.nn as nn
import torchvision.models as models
from Model_block import ResBlock 
from Model_block import DisBlock 
from show_data import showMid
from Par_train import USE_GPU,viewCount,viewMidPic
from Model_Univers import *
device = torch.device("cuda")
#define ResBlock

class dis_net(nn.Module):
    def __init__(self, num_classes = 1024,use_gpu=USE_GPU, preTrain = False,stride = 1,num_samples = 10,resnetNum = 50):
        super(dis_net, self).__init__() 

        if resnetNum == 18:
            self.model= models.resnet34(pretrained=preTrain)
        elif resnetNum == 34:
            self.model= models.resnet34(pretrained=preTrain)
        elif resnetNum == 50:
            self.model= models.resnet34(pretrained=preTrain)
        else:
            print("Error: only support resnet 18/34/50, please varify your settings!")
            exit()
        # avg pooling to global pooling
        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)
        self.model.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
           
        #    self.model.layer4[0].downsample[0].stride = (1,1)
        #    self.model.layer4[0].conv2.stride = (1,1)
        #    self.model.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        
        #models.resnet18(pretrained=False)
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                               dilate=replace_stride_with_dilation[0])
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        #                               dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)        
        #  x = self.conv1(x)
        #  x = self.bn1(x)
        #  x = self.relu(x)
        #  x = self.maxpool(x)

        #  x = self.layer1(x)
        #  x = self.layer2(x)
        #  x = self.layer3(x)
        #  x = self.layer4(x)

        #  x = self.avgpool(x)
        #  x = torch.flatten(x, 1)
        #  x = self.fc(x)

        self.dislayer =  DisBlock(in_channel=512, num_classes=num_classes,num_samples = num_samples,use_gpu=use_gpu)
    
    def forward(self, x):
        #test
        if viewMidPic:
            outfolder = "modelPic" + str(viewCount)
            oripic = x
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if viewMidPic:
            beforel1 = x
        x = self.model.layer1(x)
        x = self.model.layer2(x)            
        if viewMidPic:
            beforel3 = x
        x = self.model.layer3(x)
        x = self.model.layer4(x)       
        
        if viewMidPic:
            beforeout = x
        out = self.dislayer(x)
        #test
        if viewMidPic:
            showMid(oripic[0], outfolder +"/Ori")
            showMid(beforel1[0] ,outfolder +"/Afterl0")
            showMid(beforel3[0], outfolder +"/Afterl2")
            showMid(beforeout[0], outfolder +"/Afterl4")
        
        return out

class tow_view_net(nn.Module):
    def __init__(self, share = False,num_classes = 1024,use_gpu=USE_GPU, preTrain = False,resnetNum = 50):
        super(tow_view_net, self).__init__() 
        self.model_1 = dis_net(num_classes = num_classes,use_gpu=use_gpu, preTrain = preTrain,resnetNum = resnetNum)
        if share:
            self.model_2 = self.model_1
        else:
            self.model_2 = dis_net(num_classes = num_classes,use_gpu=use_gpu, preTrain = preTrain,resnetNum = resnetNum)

    def forward(self, x1 = None,x2 = None):
        if x1 == None:
            y1 = None
        else:
            y1 = self.model_1(x1)
        if x2 == None:
            y2 = None
        else:
            y2 = self.model_2(x2)
        return y1,y2   
        
class three_view_net(nn.Module):
    def __init__(self, share = False,class_num = 1024,use_gpu=USE_GPU, preTrain = False,resnetNum = 50):
        super(three_view_net, self).__init__() 

        self.model_1 =  ft_net(class_num, stride = 1,pool = 'avg')
        self.model_2 =  ft_net(class_num, stride = 1,pool = 'avg')
        self.model_3 = self.model_1
        self.classifier = None#ClassBlock(2048, 701, 0.75)
        self.disblock_1 = DisBlock(in_channel=2048,  out_channel = 2048,num_samples = 20,use_gpu=USE_GPU)
        self.disblock_2 = DisBlock(in_channel=2048,  out_channel = 2048,num_samples = 20,use_gpu=USE_GPU)
        self.disblock_3 = self.disblock_1


        #self.model_1 = dis_net(num_classes = num_classes,use_gpu=use_gpu, preTrain = preTrain,resnetNum = resnetNum)
        #if share:
        #    self.model_2 = self.model_1
        #else:
        #    self.model_2 = dis_net(num_classes = num_classes,use_gpu=use_gpu, preTrain = preTrain,resnetNum = resnetNum)
    def change(self):
        self.classifier = None
        self.disblock_1 = DisBlock(in_channel=2048, out_channel = 2048,num_samples = 20,use_gpu=USE_GPU)
        self.disblock_2 = DisBlock(in_channel=2048, out_channel = 2048,num_samples = 20,use_gpu=USE_GPU)
        self.disblock_3 = self.disblock_1
        
    def forward(self, satellite = None, ground = None,drone = None):
        if satellite == None:
            y1 = None
        else:
            y1 = self.model_1(satellite)
            y1 = self.disblock_1(y1)
        if ground  == None:
            y2 = None
        else:
            y2 = self.model_2(ground)
            y2 = self.disblock_2(y2)
        if drone == None:
            y3 = None
        else:
            y3 = self.model_3(drone)
            y3 = self.disblock_3(y3)
        return y1,y2,y3   
                
        
        
        
    
