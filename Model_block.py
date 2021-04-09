# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:44:12 2021

@author: Jinda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(ResBlock, self).__init__()
        
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding = 1, bias=False),
            #3*3 kernel Conv ,size remain ,outchannel : Len - ken + 1 +2 pad / stride
            nn.BatchNorm2d(out_channel),
            #gui yi hua, tongdaoshu he yige bilv 
            nn.ReLU(inplace=True),
            #if true  x=x*2 else y=x*2
            nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channel)
            )
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel)
                )
    def forward(self, x):
        out = self.left(x)
        out = out +self.shortcut(x)
        out = self.relu(out)
        return out 


class DisBlock(nn.Module):
    ''' return tuple(avg,dis,smaples)
        avg is the traditional features
        dis is the distribution features
        samples contains numclass=100 sample features
    '''
    def __init__(self, in_channel = 512, out_channel = 512, stride = 2,num_samples = 100,use_gpu=False):
        #define self parameters
        super(DisBlock, self).__init__()
        self.use_gpu = use_gpu
        self.inchannel = in_channel
        self.num_samples = num_samples
        
        #define network
        '''
        self.avgLayer = self.make_layer(ResBlock, channels = out_channel, num_blocks = 2, stride=stride)
        self.afc = nn.Linear(out_channel*9, num_classes)
        self.avg_poolLayer = nn.AvgPool2d(4)

        self.disLayer = self.make_layer(ResBlock, channels = out_channel, num_blocks = 2, stride=stride)
        self.dfc = nn.Linear(out_channel*9, num_classes)
        self.dis_poolLayer = nn.AvgPool2d(4)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.dfc = nn.Linear(out_channel*9, num_classes)
        self.dis_poolLayer = nn.AvgPool2d(4)
        '''
        self.disLayer = self.make_layer(ResBlock, channels = out_channel, num_blocks = 2, stride=stride)
        

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        inchannel = self.inchannel
        for stride in strides:
            layers.append(block(inchannel,channels,stride))
            inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        '''
        avg = self.avgLayer(x)
        avg = self.avg_poolLayer(avg)
        avg = torch.flatten(avg, 1)
        avg = self.afc(avg)
        avg = self.relu(avg)
        avg = avg.reshape(-1,self.num_classes)
        
        dis = self.dfc(dis)
        dis = self.relu(dis)
        dis = dis.reshape(-1,self.num_classes)
        #dis yao softmax
        # dis = dis + torch.ones(dis.size())
        '''

        #now cal mean
        avg = x.mean([-2,-1])
        #and dis
        disx = Variable(x.detach()) 
        dis = self.disLayer(disx)
        dis = dis.mean([-2,-1])
        
        samples = self.getSamples(avg,dis)
        
        return avg,dis,samples
        
    #ResNet    
    #    out = F.avg_pool2d(out,4)    
        #F.avg_pool1d(input, kernel_size)
    #    out = out.view(out.size(0),-1)
        #view是改变tensor的形状，view中的-1是自适应的调整 -> 512/4 = 128 -> 128副图像，view成128个向量
    #    out  = self.fc(out)
    #disNet:Now out = (512副均值图像，512副方差图像，【N*512副samples图像】) / -> 改成 长度为numclasss的向量 
    #           out = (长度为numclasss的均值向量，方差向量，【N*samples向量】) / -> 改成 长度为numclasss的向量 
        
    def getSamples(self,avgOri,dis):
        samples = []
        avg = Variable(avgOri.detach())
        
        for i in range(self.num_samples):
            #sample from N(1,0)
            torchSize = tuple(avg.size())
            randomgauss = np.random.randn(*torchSize)
            randomgauss = torch.from_numpy(randomgauss)
            if self.use_gpu== True :
                randomgauss = randomgauss.to("cuda")
            #e mult dis
            distribution =dis * randomgauss
            #dis plus avg
            sample = avg + distribution
            samples.append(sample.float())
        
        return samples
