# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:25:04 2021

@author: Jinda
"""

from __future__ import print_function, division
import Par_train as para
from Model_distributeNet import three_view_net, three_view_resNet, two_view_net
import numpy as np
import faiss
import torch
from Data_presolveing import getsatedatasets, getgrounddatasets, getdronedatasets,getpolarsatedatasets 
version =  torch.__version__
MODELPATH = "./model/three_view/net_070.pth"
load = False

def savedata(path,variable):
    torch.save(variable,path)

def cal(model):
    #print("test_vector_sate")
    #slabelsets, svectorsets = getsatedatasets(model)
    #torch.save(slabelsets,"test_vector/satelabel.dt")
    #torch.save(svectorsets,"test_vector/satevector.dt")
    print("test_vector_polar")
    slabelsets, svectorsets = getpolarsatedatasets(model)
    torch.save(slabelsets,"test_vector/polarsatelabel.dt")
    torch.save(svectorsets,"test_vector/polarsatevector.dt")
    #print("test_vector_drone")
    #slabelsets, svectorsets = getdronedatasets(model)
    #torch.save(slabelsets,"test_vector/dronelabel.dt")
    #torch.save(svectorsets,"test_vector/dronevector.dt")
    print("test_vector_ground")
    slabelsets, svectorsets = getgrounddatasets(model)
    torch.save(slabelsets,"test_vector/groundlabel.dt")
    torch.save(svectorsets,"test_vector/groundvector.dt")

def test_three(search,datasets1,datasets2):
    if search == 'satellite':
        slabelsets = torch.load("test_vector/satelabel.dt")
        svectorsets = torch.load("test_vector/satevector.dt")
    elif search == 'polar':
        slabelsets = torch.load("test_vector/polarsatelabel.dt")
        svectorsets = torch.load("test_vector/polarsatevector.dt")
    elif search == 'drone':
        slabelsets = torch.load("test_vector/dronelabel.dt")
        svectorsets = torch.load("test_vector/dronevector.dt")
    else:
        slabelsets = torch.load("test_vector/groundlabel.dt")
        svectorsets = torch.load("test_vector/groundvector.dt")

    if datasets1 == 'satellite':
        dlabelsets = torch.load("test_vector/satelabel.dt")
        dvectorsets = torch.load("test_vector/satevector.dt")
    elif datasets1 == 'polar':
        dlabelsets = torch.load("test_vector/polarsatelabel.dt")
        dvectorsets = torch.load("test_vector/polarsatevector.dt")
    elif datasets1 == 'drone':
        dlabelsets = torch.load("test_vector/dronelabel.dt")
        dvectorsets = torch.load("test_vector/dronevector.dt")
    else:
        dlabelsets = torch.load("test_vector/groundlabel.dt")
        dvectorsets = torch.load("test_vector/groundvector.dt")

    if datasets2 == 'satellite':
        dlabelsets2 = torch.load("test_vector/satelabel.dt")
        dvectorsets2 = torch.load("test_vector/satevector.dt")
    elif datasets2 == 'polar':
        dlabelsets2 = torch.load("test_vector/polarsatelabel.dt")
        dvectorsets2 = torch.load("test_vector/polarsatevector.dt")
    elif datasets2 == 'drone':
        dlabelsets2 = torch.load("test_vector/dronelabel.dt")
        dvectorsets2 = torch.load("test_vector/dronevector.dt")
    else:
        dlabelsets2 = torch.load("test_vector/groundlabel.dt")
        dvectorsets2 = torch.load("test_vector/groundvector.dt")
    svectorsets = np.array(svectorsets)
    slabelsets =  np.array(slabelsets)
    dvectorsets2 = np.array(dvectorsets2)
    dlabelsets2 =  np.array(dlabelsets2)
    dvectorsets = np.array(dvectorsets)
    dlabelsets =  np.array(dlabelsets)

    
    print("feature Size:%d"%len(svectorsets[0]))

    index = faiss.IndexFlatL2(len(svectorsets[0]))
    index.add(dvectorsets)
    
    index2 = faiss.IndexFlatL2(len(svectorsets[0]))
    index2.add(dvectorsets2)
    
    total = int(len(slabelsets) / 100) 
    D, I = index.search(svectorsets, total)
    D2, I2 = index2.search(svectorsets, total)
     
    top_one = 0
    top_ten = 0
    top_P = 0
    test_logfile = open("logfile/" + search + "->" + datasets1 + ',' + datasets2 + ".txt","w")

    print("testing-------------")
    for i,result in enumerate(I):
        #distance and index result
        index = result
        sfolder = slabelsets[i]
        print('Ground picture: %d '%i,file=test_logfile)
        
        for ii,j in enumerate(index):
            print('searchFolder:%s |TargetFolder:%s | Loss:%f'%(sfolder,dlabelsets[j],D[i][ii]),file=test_logfile)
            if dlabelsets[j] == sfolder:
                if ii==0:
                    top_one+=1
                if ii<10:
                    top_ten+=1
                top_P+=1

        print('-'*20,file=test_logfile)
    total = i+1  
    #print('total: %d | topOne%f  |  topThree%f  |  topFive%f  |  topTen%f  '%(total,top_one/total,top_three/total,top_five/total,top_ten/total))
    print('total: %d | topOne:%f  |  topTen:%f  |  topPerc1:%f  '%(total,top_one/total,top_ten/total,top_P/total))
            
def test_two(search,datasets):
    if search == 'satellite':
        slabelsets = torch.load("test_vector/satelabel.dt")
        svectorsets = torch.load("test_vector/satevector.dt")
    elif search == 'polar':
        slabelsets = torch.load("test_vector/polarsatelabel.dt")
        svectorsets = torch.load("test_vector/polarsatevector.dt")
    elif search == 'drone':
        slabelsets = torch.load("test_vector/dronelabel.dt")
        svectorsets = torch.load("test_vector/dronevector.dt")
    else:
        slabelsets = torch.load("test_vector/groundlabel.dt")
        svectorsets = torch.load("test_vector/groundvector.dt")

    if datasets == 'satellite':
        dlabelsets = torch.load("test_vector/satelabel.dt")
        dvectorsets = torch.load("test_vector/satevector.dt")
    elif datasets == 'polar':
        dlabelsets = torch.load("test_vector/polarsatelabel.dt")
        dvectorsets = torch.load("test_vector/polarsatevector.dt")
    elif datasets == 'drone':
        dlabelsets = torch.load("test_vector/dronelabel.dt")
        dvectorsets = torch.load("test_vector/dronevector.dt")
    else:
        dlabelsets = torch.load("test_vector/groundlabel.dt")
        dvectorsets = torch.load("test_vector/groundvector.dt")
    svectorsets = np.array(svectorsets)
    slabelsets =  np.array(slabelsets)

    dvectorsets = np.array(dvectorsets)
    dlabelsets =  np.array(dlabelsets)

    
    print("feature Size:%d"%len(svectorsets[0]))

    index = faiss.IndexFlatL2(len(svectorsets[0]))
    index.add(dvectorsets)
    
    total = int(len(slabelsets) / 100) 
    D, I = index.search(svectorsets, total)
     
    top_one = 0
    top_ten = 0
    top_P = 0
    test_logfile = open("logfile/" + search + "_2_" + datasets + ".txt","w")

    print("testing-------------")
    for i,result in enumerate(I):
        #distance and index result
        index = result
        sfolder = slabelsets[i]
        print('Ground picture: %d '%i,file=test_logfile)
        
        for ii,j in enumerate(index):
            print('searchFolder:%s |TargetFolder:%s | Loss:%f'%(sfolder,dlabelsets[j],D[i][ii]),file=test_logfile)
            if dlabelsets[j] == sfolder:
                if ii==0:
                    top_one+=1
                if ii<10:
                    top_ten+=1

                top_P+=1
        print('-'*20,file=test_logfile)
    total = i+1  
    #print('total: %d | topOne%f  |  topThree%f  |  topFive%f  |  topTen%f  '%(total,top_one/total,top_three/total,top_five/total,top_ten/total))
    print('total: %d | topOne:%f  |  topTen:%f  |  topPerc1:%f  '%(total,top_one/total,top_ten/total,top_P/total))
            

def test(model):
    if load == False:
        #dictory
        print("-"*10)
        print("get polarsatellite database")
        slabelsets, svectorsets = getsatedatasets(model)
        print("drone datasets Size:%d"%len(svectorsets))
        print("-"*10)
        ###search
        print("-"*10)
        print("get drone database")
        glabelsets, gvectorsets = getdronedatasets(model)
        print("ground datasets Size:%d"%len(svectorsets))
        print("-"*10)
        satesets = np.array(svectorsets)
        slabelsets =  np.array(slabelsets)
        
        groundsets = np.array(gvectorsets)
        glabelsets =  np.array(glabelsets)
    
        
        print("feature Size:%d"%len(satesets[0]))

        index = faiss.IndexFlatL2(len(satesets[0]))
        index.add(satesets)
        
        total = int(len(glabelsets) / 100) 
        print(total)
        D, I = index.search(groundsets, total)
    
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
        
       
    print(total)
    top_three = 0
    top_one = 0
    top_five = 0
    top_ten = 0
    top_P = 0

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
                if ii==0:
                    top_one+=1
                if ii<10:
                    top_ten+=1

                top_P+=1
       
        print('-'*20,file=test_logfile)
    total = i+1  
    #print('total: %d | topOne%f  |  topThree%f  |  topFive%f  |  topTen%f  '%(total,top_one/total,top_three/total,top_five/total,top_ten/total))
    print('total: %d | topOne:%f  |  topTen:%f  |  topPerc1:%f  '%(total,top_one/total,top_ten/total,top_P/total))
            
def getNoisePair(search,datasets,percentage):
    if search == 'satellite':
        slabelsets = torch.load("test_vector/satelabel.dt")
        svectorsets = torch.load("test_vector/satevector.dt")
    elif search == 'polar':
        slabelsets = torch.load("test_vector/polarsatelabel.dt")
        svectorsets = torch.load("test_vector/polarsatevector.dt")
    elif search == 'drone':
        slabelsets = torch.load("test_vector/dronelabel.dt")
        svectorsets = torch.load("test_vector/dronevector.dt")
    else:
        slabelsets = torch.load("test_vector/groundlabel.dt")
        svectorsets = torch.load("test_vector/groundvector.dt")

    if datasets == 'satellite':
        dlabelsets = torch.load("test_vector/satelabel.dt")
        dvectorsets = torch.load("test_vector/satevector.dt")
    elif datasets == 'polar':
        dlabelsets = torch.load("test_vector/polarsatelabel.dt")
        dvectorsets = torch.load("test_vector/polarsatevector.dt")
    elif datasets == 'drone':
        dlabelsets = torch.load("test_vector/dronelabel.dt")
        dvectorsets = torch.load("test_vector/dronevector.dt")
    else:
        dlabelsets = torch.load("test_vector/groundlabel.dt")
        dvectorsets = torch.load("test_vector/groundvector.dt")
    svectorsets = np.array(svectorsets)
    slabelsets =  np.array(slabelsets)

    dvectorsets = np.array(dvectorsets)
    dlabelsets =  np.array(dlabelsets)

    
    print("feature Size:%d"%len(svectorsets[0]))


    index = faiss.IndexFlatL2(len(svectorsets[0]))
    index.add(dvectorsets)
    
    total = int(len(slabelsets) / 100) 
    number = int(total*percentage*100)

    D, I = index.search(svectorsets, total) 
    
    noisePair_logfile = open("logfile/NoisePair" + search + "_2_" + datasets + ".txt","w")

    print("testing-------------")
    n = 0
    while n < total:
        for i,result in enumerate(I):
            #distance and index result
            index = result
            sfolder = slabelsets[i]
            j = index[n]
            if dlabelsets[j] != sfolder:
                print('%s %s'%(sfolder,dlabelsets[j]),file=noisePair_logfile)
                number-=1
            if number < 0:
                break
        if number < 0 :
            break
                   
if __name__ == '__main__':
    views = 1
    device = torch.device("cpu")
    use_gpu =para.USE_GPU
    if use_gpu:
        device = torch.device("cuda")
        use_gpu = torch.cuda.is_available()
    if para.IS_DIS_Net:
        if views == 3:
            model = three_view_net(use_gpu=para.USE_GPU,resnetNum=18).to(device)
        else :   
            model = two_view_net(use_gpu=para.USE_GPU,resnetNum=18).to(device)
    else:
        model = three_view_resNet()
    model.load_state_dict(torch.load(MODELPATH))
    cal(model)
    #test_two(search='polar',datasets='drone')
    #test_two(search='drone',datasets='polar')
    #test_two(search='polar',datasets='ground')
    test_two(search='ground',datasets='polar')
    #getNoisePair(search='drone',datasets='satellite',percentage= 0.2)
    #test(model)