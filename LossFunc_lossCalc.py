# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:06:01 2021

@author: Jinda
"""
import torch.nn as nn
import math

triplet_loss = nn.TripletMarginLoss(margin=1, p=2)
#lp_loss l2范数，应该是计算为欧氏距离
#torch.nn.TripletMarginWithDistanceLoss(*, distance_function=None, margin=1.0, swap=False, reduction='mean')
#这个是可以自己指定距离函数的，distance_function
mse_loss = nn.MSELoss()
#torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
#torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

def TripletLoss(anchor,positive,negative):
    
    return triplet_loss(anchor, positive, negative)
       
def SampleLoss(samples,target):
    #[N*samples向量] and meansTarget
    totalLoss = 0.0
    for i in range(len(samples)):
        totalLoss += mse_loss(samples[i],target)
    return totalLoss/(i+1)

def FeaturesLoss(manchor,sanchor,mpositive,spositvie,mnegative,K):
    #anchor -> (mean,[sample])
    totalLoss = 0.0
    totalLoss += TripletLoss(manchor, mpositive, mnegative)
    totalLoss += K * SampleLoss(sanchor, mpositive)
    totalLoss += K * SampleLoss(spositvie, manchor)
    return totalLoss
    
def FeaturesLossWithoutSample(manchor,mpositive,mnegative):
    #anchor -> (mean,[sample])
    totalLoss = 0.0
    totalLoss += TripletLoss(manchor, mpositive, mnegative)
    return totalLoss

def UncertaintyLoss(disanchor):
    
    m = len(disanchor)
    n = len(disanchor[0])
    #each vector has a diagno matrix
    sigma =0.0
    sumD = 0.0
    for i in range(m):
        
        for j in range(n):
            sumD += disanchor[i][j]*disanchor[i][j]
        if sumD != 0 :
            sigma += math.log(sumD)
    totalLoss = m/2*(math.log(math.pi) + 1) + 1/2 * sigma
    
    return totalLoss
    
# out = (长度为numclasss的均值向量，方差向量，) 
# result = (out1,out2,out3,out4,out5,out6)