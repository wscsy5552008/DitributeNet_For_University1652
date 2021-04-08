# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:37:14 2021

@author: Jinda
"""

import torch
import torch.nn as nn
import torchvision.models as models
from Model_block import ResBlock 
from Model_block import DisBlock 
from show_data import showMid

device = torch.device("cuda")
#define ResBlock
viewMidPic = False
class DisNet(nn.Module):
    def __init__(self, num_classes = 128,ccuda=False):
        super(DisNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer1 = self.make_layer(ResBlock, channels = 64, num_blocks = 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, channels = 128, num_blocks = 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, channels = 256, num_blocks = 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, channels = 512, num_blocks = 2, stride=2)
        #self.fc = nn.Linear(512, num_classes)
        #disNet
        self.layer5 = DisBlock(in_channel=512,out_channel=512,num_classes=num_classes,cuda=ccuda) 
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel,channels,stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward_one(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
            
        return out
    
    def forward(self, x1,x2=None,x3=None,x4=None,x5=None,x6=None):
        if x2==None :#only one
            return self.forward_one(x1)
        elif x5==None:#for ground & satelite
            return self.forward_one(x1),self.forward_one(x2),self.forward_one(x3),self.forward_one(x4)
        else:#for all
            y1 = self.forward_one(x1)
            y2 = self.forward_one(x2)
            y3 = self.forward_one(x3)
            y4 = self.forward_one(x4)
            y5 = self.forward_one(x5)
            y6 = self.forward_one(x6)
            return y1,y2,y3,y4,y5,y6
        
class PreTrainDisNet(nn.Module):
    def __init__(self, num_classes = 1024,ccuda=False):
        super(PreTrainDisNet, self).__init__()
        self.gmodel= models.resnet18(pretrained=False)
        self.sdmodel = self.gmodel
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
        
        self.gdislayer =  DisBlock(in_channel=512, num_classes=num_classes,cuda=ccuda)
        self.sddislayer =  DisBlock(in_channel=512, num_classes=num_classes,cuda=ccuda)
    
    def forward_one(self, x):
        #test
        if viewMidPic:
            oripic = x
        #test
        if self.ground:
            
            out = self.gmodel.conv1(x)
            out = self.gmodel.bn1(out)
            out = self.gmodel.relu(out)

            if viewMidPic:
                outfolder = "modelPic/ground"
                beforel1 = out
            
            out = self.gmodel.layer1(out)
            out = self.gmodel.layer2(out)
            
            if viewMidPic:
                beforel3 = out
            
            out = self.gmodel.layer3(out)
            out = self.gmodel.layer4(out)
            
            if viewMidPic:
                beforedis = out
            
            out = self.gdislayer(out)
        else:
            
            
            out = self.sdmodel.conv1(x)
            out = self.sdmodel.bn1(out)
            out = self.sdmodel.relu(out)
            
            if viewMidPic:
                outfolder = "modelPic/satellite"
                beforel1 = out

            out = self.sdmodel.layer1(out)
            out = self.sdmodel.layer2(out)
            
            if viewMidPic:
                beforel3 = out
            
            out = self.sdmodel.layer3(out)
            out = self.sdmodel.layer4(out)

            if viewMidPic:
                beforedis = out
            
            out = self.sddislayer(out)
            
        if viewMidPic:
            showMid(oripic[0], outfolder +"/Ori")
            showMid(beforel1[0] ,outfolder +"/l0")
            showMid(beforel3[0], outfolder +"/l2")
            showMid(beforedis[0], outfolder +"/l4")
        
        return out
         
    def forward(self, x1=None,x2=None,x3=None,x4=None,x5=None,x6=None):
        if x2==None and x3==None:#only for ground
            self.ground = True
            y1 = self.forward_one(x1)
            return y1
        elif x1==None and x2==None :#only for satelite
            self.ground = False
            return self.forward_one(x3)
        elif x3==None:#for ground & satelite
            self.ground = True
            y1 = self.forward_one(x1)
            self.ground = False
            y2 = self.forward_one(x2)
            return y1,y2
        elif x1 == None:#for drone and satellite
            self.ground = False
            y2 = self.forward_one(x2)
            y3 = self.forward_one(x3)
            y5 = self.forward_one(x5)
            y6 = self.forward_one(x6)
            return y2,y3,y5,y6   
        else:#for al
            self.ground = True
            y1 = self.forward_one(x1)
            y4 = self.forward_one(x4)
            self.ground = False
            y2 = self.forward_one(x2)
            y3 = self.forward_one(x3)
            y5 = self.forward_one(x5)
            y6 = self.forward_one(x6)
            return y1,y2,y3,y4,y5,y6   
        
        
        
        
    
