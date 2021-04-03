# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:25:04 2021

@author: Jinda
"""

from __future__ import print_function, division
import numpy as np
import faiss
import torch
from Data_presolveing import getsatedatasets, getgrounddatasets
version =  torch.__version__
MODELPATH = ""

def test(model):
    
    slabelsets, svectorsets = getsatedatasets(model)
    glabelsets, gvectorsets = getgrounddatasets(model)
    
    satesets = np.array(svectorsets)
    slabelsets =  np.array(slabelsets)
    
    groundsets = np.array(gvectorsets)
    glabelsets =  np.array(glabelsets)

    
    #128 dimension vector
    index = faiss.IndexFlatL2(len(satesets[0]))
    index.add(satesets)
    
    D, I = index.search(groundsets, 10)

    torch.save(I,"matchResult.rs")
    torch.save(glabelsets,"glabelSets.rs")
    torch.save(slabelsets,"sLabelSets.rs")
    
    total = len(I)  
    top_three = 0
    top_one = 0
    top_five = 0
    top_ten = 0

    test_logfile = open("logfile/testLog.txt","w")

    for i,result in enumerate(I):
        #distance and index result
        index = result
        gfolder = glabelsets[i]
        print('groundFolder:%s | TotalLoss:%f'%(gfolder,D[i]),file=test_logfile)
        
        for j in index:
            if slabelsets[j] == gfolder:
                if j==0:
                    top_one+=1
                elif j<3:
                    top_three+=1
                elif j<5:
                    top_five+=1
                else:
                    top_ten+=1
       
    print('topOne%f  |  topThree%f  |  topFive%f  |  topTen%f  '%(top_one/total,top_three/total,top_five/total,top_ten/total))
            

if __name__ == '__main__':
    model =torch.load(MODELPATH)
    test(model)