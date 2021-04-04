# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:25:04 2021

@author: Jinda
"""

from __future__ import print_function, division
from Model_distributeNet import PreTrainDisNet as DisNet
import numpy as np
import faiss
import torch
from Data_presolveing import getsatedatasets, getgrounddatasets
version =  torch.__version__
MODELPATH = "model/tri_view/net_000.pth"
load = False
def test(model):
    if load == False:
        print("get satelite database")
        print("-"*10)
        slabelsets, svectorsets = getsatedatasets(model)
        
        print("get ground database")
        print("-"*10)
        glabelsets, gvectorsets = getgrounddatasets(model)
        
        satesets = np.array(svectorsets)
        slabelsets =  np.array(slabelsets)
        
        groundsets = np.array(gvectorsets)
        glabelsets =  np.array(glabelsets)
    
        
        #128 dimension vector
        index = faiss.IndexFlatL2(len(satesets[0]))
        index.add(satesets)
        
        D, I = index.search(groundsets, 10)
    
        print("saving database")
        print("-"*10)
        torch.save(D,"database/matchLoss.rs")
        torch.save(I,"database/matchResult.rs")
        torch.save(glabelsets,"database/glabelSets.rs")
        torch.save(slabelsets,"database/sLabelSets.rs")
    else:
        D = torch.load("database/matchLoss.rs")
        I = torch.load("database/matchResult.rs")
        glabelsets = torch.load("database/glabelSets.rs")
        slabelsets = torch.load("database/sLabelSets.rs")
        
        
    total = len(I)  
    top_three = 0
    top_one = 0
    top_five = 0
    top_ten = 0

    test_logfile = open("logfile/testLog.txt","w")

    print("testing-------------")
    for i,result in enumerate(I):
        #distance and index result
        index = result
        gfolder = glabelsets[i]
        print('Ground picture: %d '%i,file=test_logfile)
        
        for ii,j in enumerate(index):
            print('groundFolder:%s |TargetFolde:%s | Loss:%f'%(gfolder,slabelsets[j],D[i][ii]),file=test_logfile)
            if slabelsets[j] == gfolder:
                if j==0:
                    top_one+=1
                if j<3:
                    top_three+=1
                if j<5:
                    top_five+=1

                top_ten+=1
       
        print('-'*20,file=test_logfile)
    print('topOne%f  |  topThree%f  |  topFive%f  |  topTen%f  '%(top_one/total,top_three/total,top_five/total,top_ten/total))
            

if __name__ == '__main__':
    model = DisNet()
    model.load_state_dict(torch.load(MODELPATH))
    test(model)