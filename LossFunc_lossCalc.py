# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:06:01 2021

@author: Jinda
"""
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from Par_train import USE_GPU
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
    """

    def __init__(self, margin = 10.0,use_gpu=USE_GPU):
        super(FrobeniusTriLoss, self).__init__()
        self.margin = margin
        self.use_gpu = use_gpu
         
    def forward(self, anchor, posi, nege = None, alpha = 0.01):
        
        f1 = torch.norm(anchor - posi)
        if nege != None:
            f2 = torch.norm(anchor - nege)
            m = (f1-f2 + self.margin)
        else :
            m = f1

        t0 = torch.zeros(1) + math.e
        
        if self.use_gpu == True:
            t0 =  Variable(t0.cuda().detach())
           
        #print("FrobeniusLoss: %f"%m)
        if m < 0 :
            m = torch.zeros(1,dtype=float)
            m.requires_grad = True
        
        t1 = torch.pow(t0, alpha * m)
        
        return torch.log(t1)
    
TriFrobeniusLoss = FrobeniusTriLoss()

def SampleLoss(samples,target):
    #[N*samples向量] and meansTarget
    totalLoss = torch.ones(1,dtype = target.dtype,requires_grad=True)
    if USE_GPU:
        totalLoss = Variable(totalLoss.cuda().detach())
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
    totalLoss = TriFrobeniusLoss(manchor,mpositive,mnegative, 1)
    #print("FeatureLoss:%f"%totalLoss)
    
    #考虑以下anchor应该不能靠过来
    ma,mp = Variable(manchor.detach()),Variable(mpositive.detach())
    
    #totalLoss = totalLoss + K * SampleLoss(sanchor, ma)
    #totalLoss = totalLoss + K * SampleLoss(sanchor, mp)
    
    #print("FeatureLossForSample:%f"%totalLoss)
    return totalLoss
    
def FeaturesLossWithoutSample(manchor,mpositive,mnegative):
    #anchor -> (mean,[sample])
    totalLoss = TriFrobeniusLoss(manchor,mpositive,mnegative, 0.01) #CrossTowviewLoss(manchor, mpositive, 1) + CrossTowviewLoss(manchor, mnegative, 0)
    return totalLoss

def UncertaintyLoss(disanchor,use_gpu = USE_GPU):

    batch_size = len(disanchor)
    n = len(disanchor[0])
    #each vector has a diagno matrix
    
    #1/2 log(  2 pi e squre)
    #sigma = torch.zeros(size=(1,1), dtype= float
        
    sumD = torch.zeros(size=(1,1), dtype= disanchor.dtype, requires_grad=True)
    if use_gpu == True:
        sumD = Variable(sumD.cuda().detach())
        
    for i in range(batch_size):   
        #batch 
        for j in range(n):
            if (disanchor[i][j] != 0 and disanchor[i][j] != 1):
                #这是按照原来论文设计的 
                sumD = sumD + torch.log(disanchor[i][j])
                #以下是改进的，希望每个disanchor都变成1
            elif disanchor[i][j] == 0:
                sumD = sumD + torch.log(2-disanchor[i][j])

            
    #这是按照原来论文设计的 totalLoss = batch_size * n/2 *(math.log(2*math.pi) + 1) + 1/2 * sumD
    #以下是我修改的
    return sumD