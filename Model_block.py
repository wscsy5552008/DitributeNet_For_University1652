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
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel)
                )
    def forward(self, x):
        out = self.left(x)
        out = out +self.shortcut(x)
        out = F.relu(out)
        return out 


class DisBlock(nn.Module):
    def __init__(self, in_channel = 256, out_channel = 512, stride = 1,num_classes = 128,cuda=False):
        self.cuda = cuda
        super(DisBlock, self).__init__()
        self.inchannel = in_channel
        self.num_classes = num_classes
        self.avgLayer = self.make_layer(ResBlock, channels = out_channel, num_blocks = 2, stride=2)
        self.avg_poolLayer = nn.AvgPool2d(4,stride = 1)
        self.avg_relu = nn.ReLU()
        
        self.disLayer = self.make_layer(ResBlock, channels = out_channel, num_blocks = 2, stride=2)
        self.afc = nn.Linear(out_channel*9, num_classes)
        self.dfc = nn.Linear(out_channel*9, num_classes)
        self.dis_poolLayer = nn.AvgPool2d(4,stride = 1)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=0)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        inchannel = self.inchannel
        for stride in strides:
            layers.append(block(inchannel,channels,stride))
            inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        avg = self.avgLayer(x)
        avg = self.avg_poolLayer(x)
        avg = torch.flatten(avg, 1)
        avg = self.afc(avg)
        
        
        #avg = avg.view(avg.size(0),-1)
        #avg = self.fc(avg)
        avg = self.relu(avg)
        avg = avg.reshape(-1,self.num_classes)
        
        disx = Variable(x.detach())
        dis = self.disLayer(disx)
        dis = self.dis_poolLayer(dis)
        
        dis = torch.flatten(dis, 1)
        dis = self.dfc(dis)
        
        #dis = dis.view(dis.size(0),-1)
        #dis = self.fc(dis)
        dis = self.relu(dis)
        dis = dis.reshape(-1,self.num_classes)
        #dis yao softmax
        #dis = self.softmax(dis)
        
        return avg,dis,self.getSamples(avg,dis)
        
    #ResNet    
    #    out = F.avg_pool2d(out,4)    
        #F.avg_pool1d(input, kernel_size)
    #    out = out.view(out.size(0),-1)
        #view是改变tensor的形状，view中的-1是自适应的调整 -> 512/4 = 128 -> 128副图像，view成128个向量
    #    out  = self.fc(out)
    #disNet:Now out = (512副均值图像，512副方差图像，【N*512副samples图像】) / -> 改成 长度为numclasss的向量 
    #           out = (长度为numclasss的均值向量，方差向量，【N*samples向量】) / -> 改成 长度为numclasss的向量 
        
    def getSamples(self,avgOri,dis,size = 10):
        samples = []
        avg = Variable(avgOri.detach())
        for i in range(size):
            #sample from N(1,0)
            torchSize = tuple(avg.size())
            e = np.random.randn(*torchSize)
            e = torch.from_numpy(e)
            if self.cuda== True :
                e = e.to("cuda")
            #e mult dis
            distribute = e * dis
            #dis plus avg
            sample = distribute + avg
            samples.append(sample.float())
        #return sample
        return samples
