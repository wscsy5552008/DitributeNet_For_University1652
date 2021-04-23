# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:37:14 2021

@author: Jinda
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from show_data import showMid
from Par_train import USE_GPU,viewCount,viewMidPic

class MiniBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(MiniBlock, self).__init__()
        
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

class ResBlock(nn.Module):

    def __init__(self, in_channel,num_classes = 2048, out_channel = 512, stride = 2):
        #define self parameters
        super(DisBlock, self).__init__()
        self.inchannel = in_channel
        self.Layer = self.make_layer(MiniBlock, channels = out_channel, num_blocks = 2, stride=stride)
        self.fc = nn.Linear(out_channel, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        inchannel = self.inchannel
        for stride in strides:
            layers.append(block(inchannel,channels,stride))
            inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.Layer(x)
        x = self.fc(x)
        
        return x
        
class DisBlock(nn.Module):
    ''' return tuple(avg,dis,smaples)
        avg is the traditional features
        dis is the distribution features
        samples contains numclass=10 sample features
    '''
    def __init__(self, in_channel, out_channel = 2048,num_classes = 2048, stride = 2,num_samples = 10,use_gpu=False):
        #define self parameters
        super(DisBlock, self).__init__()
        self.use_gpu = use_gpu
        self.inchannel = in_channel
        self.num_samples = num_samples
        self.softmax = nn.Softmax(dim=0)

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
        self.disLayer = self.make_layer(MiniBlock, channels = out_channel, num_blocks = 2, stride=stride)
        self.avgLayer = self.make_layer(MiniBlock, channels = out_channel, num_blocks = 2, stride=stride)
        #self.dfc = nn.Linear(out_channel, num_classes)
        #self.afc = nn.Linear(out_channel, num_classes)

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
        avg = self.avgLayer(x)
        #avg = self.afc(avg)
        avg = avg.mean([-2,-1])
        #and dis
        disx = Variable(x.detach()) 
        dis = self.disLayer(disx)
        #dis = self.dfc(dis)
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
                if randomgauss.dtype != dis.dtype:
                    randomgauss = randomgauss.to(dis.dtype)
            #e mult dis
            distribution =dis * randomgauss
            #dis plus avg
            sample = avg + distribution
            samples.append(sample.float())
        
        return samples

class res_net_without_fc(nn.Module):
    def __init__(self,  preTrain = False,resnetNum = 50):
        super(res_net_without_fc, self).__init__() 

        if resnetNum == 18:
            self.model= models.resnet18(pretrained=preTrain)
        elif resnetNum == 34:
            self.model= models.resnet34(pretrained=preTrain)
        elif resnetNum == 50:
            self.model= models.resnet50(pretrained=preTrain)
        else:
            print("Error: only support resnet 18/34/50, please varify your settings!")
            exit()
    
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
        out = self.model.layer4(x)       
        
        if viewMidPic:
            beforeout = x
            
        if viewMidPic:
            showMid(oripic[0], outfolder +"/Ori")
            showMid(beforel1[0] ,outfolder +"/Afterl0")
            showMid(beforel3[0], outfolder +"/Afterl2")
            showMid(beforeout[0], outfolder +"/Afterl4")
        
        return out

class two_view_net(nn.Module):
    def __init__(self, share = False,num_samples = 10,use_gpu=USE_GPU, preTrain = True,resnetNum = 50):
        super(two_view_net, self).__init__() 
        #<50 512
        #>=50 2048
        if resnetNum < 50 :
            res_out_chan = 512
        else:
            res_out_chan = 2048

        self.model_1 = res_net_without_fc(preTrain = preTrain,resnetNum = resnetNum)
        self.disblock_1 = DisBlock(in_channel=res_out_chan,  out_channel = 512,num_samples = num_samples,use_gpu=use_gpu)
        if share:
            self.model_2 = self.model_1
            self.disblock_2 = self.disblock_1
        else:
            self.model_2 = res_net_without_fc(preTrain = preTrain,resnetNum = resnetNum)
            self.disblock_2 = DisBlock(in_channel=res_out_chan,  out_channel = 512,num_samples = 10,use_gpu=use_gpu)

    def forward(self, satellite = None, ground = None,drone = None):
        if satellite == None:
            y1 = None
        else:
            y1 = self.model_1(satellite)
            y1 = self.disblock_1(y1)
        if ground == None:
            y2 = None
        else:
            y2 = self.model_2(ground)
            y2 = self.disblock_2(y2)
        if drone == None:
            y3 = None
        else:
            y3 = self.model_2(drone)
            y3 = self.disblock_2(y3)
        return y1,y2,y3
        
class three_view_net(nn.Module):
    def __init__(self, share = False,class_num = 1024,use_gpu=USE_GPU, preTrain = True,resnetNum = 50):
        super(three_view_net, self).__init__() 

        self.model_1 = res_net_without_fc( preTrain = preTrain,resnetNum = resnetNum)
        self.model_2 = res_net_without_fc( preTrain = preTrain,resnetNum = resnetNum)
        self.model_3 = self.model_1
        #<50 512
        #>=50 2048
        if resnetNum < 50 :
            res_out_chan = 512
        else:
            res_out_chan = 2048
        self.disblock_1 = DisBlock(in_channel=res_out_chan,  out_channel = 512,num_samples = 10,use_gpu=USE_GPU)
        self.disblock_2 = DisBlock(in_channel=res_out_chan,  out_channel = 512,num_samples = 10,use_gpu=USE_GPU)
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

        
class two_view_resNet(nn.Module):
    def __init__(self, share = False,num_classes = 1024,preTrain = True,resnetNum = 34):
        super(two_view_resNet, self).__init__() 
        
        #if resnetNum <50:
        #    res_out_channel = 512
        #else:
        #    res_out_channel = 2048
            
        self.model_1 = res_net_without_fc( preTrain = preTrain,resnetNum = resnetNum)
        #self.fc1 = nn.Linear(res_out_channel, num_classes)
        if share:  
            self.model_2 = self.model_1
            #self.fc2 = self.fc1
        else:
            self.model_2 = res_net_without_fc(preTrain = preTrain,resnetNum = resnetNum)
            #self.fc2 = nn.Linear(res_out_channel, num_classes)
        self.model_3 = self.model_1
        #self.fc3 = self.fc_1
        
    def forward(self, satellite = None, ground = None,drone = None):
        if satellite == None:
            y1 = None
        else:
            y1 = self.model_1(satellite)
            y1 = y1.means([-2,-1])
            #y1 = self.fc1(y1)
        if ground  == None:
            y2 = None
        else:
            y2 = self.model_2(ground)
            y2 = y2.means([-2,-1])
            #y2 = self.fc2(y1)
        if drone == None:
            y3 = None
        else:
            y3 = self.model_3(drone)
            y3 = y3.means([-2,-1])
            #y2 = self.fc3(y1)
        return y1,y2,y3                   

class three_view_resNet(nn.Module):
    def __init__(self, share = False,num_classes = 1024,preTrain = True,resnetNum = 34):
        super(three_view_resNet, self).__init__() 
        
        if resnetNum <50:
            res_out_channel = 512
        else:
            res_out_channel = 2048
            
        self.model_1 = res_net_without_fc(preTrain = preTrain,resnetNum = resnetNum)
        self.fc1 = nn.Linear(res_out_channel, num_classes)
        self.model_2 = res_net_without_fc(preTrain = preTrain,resnetNum = resnetNum)
        self.fc2 = nn.Linear(res_out_channel, num_classes)
        self.model_3 = self.model_1
        self.fc3 = self.fc_1
        
        
    def forward(self, satellite = None, ground = None,drone = None):
        if satellite == None:
            y1 = None
        else:
            y1 = self.model_1(satellite)
            y1 = y1.means([-2,-1])
            #y1 = self.fc1(y1)
        if ground  == None:
            y2 = None
        else:
            y2 = self.model_2(ground)
            y2 = y2.means([-2,-1])
            #y2 = self.fc2(y1)
        if drone == None:
            y3 = None
        else:
            y3 = self.model_3(drone)
            y3 = y3.means([-2,-1])
            #y2 = self.fc3(y1)
        return y1,y2,y3                   
        
        
        
    
