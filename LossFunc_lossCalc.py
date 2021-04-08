# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:06:01 2021

@author: Jinda
"""
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

triplet_loss = nn.TripletMarginLoss(margin=1, p=2)
#lp_loss l2范数，应该是计算为欧氏距离
#torch.nn.TripletMarginWithDistanceLoss(*, distance_function=None, margin=1.0, swap=False, reduction='mean')
#这个是可以自己指定距离函数的，distance_function
mse_loss = nn.MSELoss()
#torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
#torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive
    
CrossTowviewLoss = ContrastiveLoss(margin=2.0)

class FrobeniusTriLoss(torch.nn.Module):
    """
    TwoView Frobenius loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin = 4.0):
        super(FrobeniusTriLoss, self).__init__()
        self.margin = margin
         
    def forward(self, anchor, posi, nege = None, alpha = 0.01):
        
        f1 = torch.norm(anchor - posi)
        if nege != None:
            f2 = torch.norm(anchor - nege)
            m = (f1-f2 + self.margin)
        else :
            m = f1
        #print("FrobeniusLoss: %f"%m)
        if m < 0 :
            return torch.zeros(1)
        
        return torch.log(1+torch.pow(torch.zeros(1) + math.e,alpha * m))
    
TriFrobeniusLoss = FrobeniusTriLoss()

class TripletUncertaintyLoss(nn.Module):
    
    def __init__(self,cuda=False):
        super(TripletUncertaintyLoss, self).__init__()
        self.cuda = cuda
        
    def forward(self,disanchor):
        m = len(disanchor)
        n = len(disanchor[0])
        #each vector has a diagno matrix
        
#      1/2 log(  2 pi e squre)
        #sigma = torch.zeros(size=(1,1), dtype= float)
        #if self.cuda == True:
        #    sigma = Variable(sigma.cuda().detach())
            
        sumD = torch.zeros(size=(1,1), dtype= float)
        if self.cuda == True:
            sumD = Variable(sumD.cuda().detach())
            
            
        for i in range(m):   
            #batch 
            for j in range(n):
                if (disanchor[i][j] != 0):
                    sumD = sumD + torch.log(disanchor[i][j])
                
        totalLoss = m* n/2 *(math.log(2*math.pi) + 1) + 1/2 * sumD
        #totalLoss = torch.log(sigma)
        if totalLoss > 0:
            return torch.log(1+totalLoss)
        else :
            return torch.ones(1)
    
# out = (长度为numclasss的均值向量，方差向量，) 
# result = (out1,out2,out3,out4,out5,out6)

#def TripletLoss(anchor,positive,negative):
    
#    return triplet_loss(anchor, positive, negative)
       
def SampleLoss(samples,target):
    #[N*samples向量] and meansTarget
    totalLoss = torch.zeros(1,dtype = float)
    #totalSample = samples[0]
    for i in range(0,len(samples)):
        #totalSample= totalSample + samples[i]
        totalLoss = totalLoss +TriFrobeniusLoss(anchor = samples[i],posi = target) #mse_loss(samples[i],target)
    return totalLoss/(i+1)
    #return CrossTowviewLoss(totalSample/(i+1),target,1)

def AvgSamples(samples):
    totalSample = samples[0]
    for i in range(1,len(samples)):
        totalSample= totalSample + samples[i]
    return totalSample/(i+1)

def FeaturesLoss(manchor,sanchor,mpositive,spositvie,mnegative,snegative,K = 1):
    #anchor -> (mean,[sample])
    totalLoss = TriFrobeniusLoss(manchor,mpositive,mnegative, 0.01)#CrossTowviewLoss(manchor, mpositive, 1) + CrossTowviewLoss(manchor, mnegative, 0)
    #print("FeatureLoss:%f"%totalLoss)
    #TripletLoss(manchor, mpositive, mnegative)
    totalLoss = totalLoss + K * SampleLoss(sanchor, manchor)
    #totalLoss = totalLoss + K * SampleLoss(sanchor, mpositive)
    #totalLoss = totalLoss + K * TriFrobeniusLoss(AvgSamples(sanchor), AvgSamples(spositvie), AvgSamples(snegative))
    #print("FeatureLossForSample:%f"%totalLoss)
    return totalLoss
    
def FeaturesLossWithoutSample(manchor,mpositive,mnegative):
    #anchor -> (mean,[sample])
    totalLoss = CrossTowviewLoss(manchor, mpositive, 1) + CrossTowviewLoss(manchor, mnegative, 0)
    return totalLoss

def UncertaintyLoss(disanchor):
    loss = TripletUncertaintyLoss()
    return loss(disanchor)
