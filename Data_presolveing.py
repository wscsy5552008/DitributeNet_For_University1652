# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:58:27 2021

@author: Jinda
"""
from Model_distributeNet import dis_net as DisNet, three_view_net
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import random
from utils_from_others.autoaugment import ImageNetPolicy
from utils_from_others.random_erasing import RandomErasing
import Par_train as para
import show_data
target_drone = 'data/train/drone'
target_ground = 'data/train/street'
target_satellite = 'data/train/satellite'
#target_root = 'data/train/google'

target_test_satellite = '../data/test/gallery_satellite'
target_test_drone = '../data/test/gallery_drone'
target_test_ground = '../data/test/gallery_street'

def pad(inp, pad = 3):
    #print(inp.size)
    h, w = inp.size
    bg = np.zeros((h+2*pad, w+2*pad, len(inp.mode)))
    bg[pad:pad+h, pad:pad+w, :] = inp
    return bg

def getsatedatasets(model):
    return gettestdatasets(model,target_test_satellite,'satellite')
    
def getgrounddatasets(model):
    return gettestdatasets(model,target_test_ground,'ground')

def getdronedatasets(model):
    return gettestdatasets(model,target_test_drone,'drone')
    
    
def gettestdatasets(model,path,view):
    
    transform1 = transforms.Compose([
        transforms.CenterCrop((para.H,para.W)), # 只能对PIL图片进行裁剪
        transforms.Pad( 10, padding_mode='edge'),
        transforms.RandomCrop((para.H,para.W)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        )
    datasets = []
    label = []
    total = len(os.listdir(path))
    foldeList =os.listdir(path)
    #random.shuffle(foldeList)
    for fi,folder_name in enumerate(foldeList,0):
        if fi >100:
            break
      
        
        folder_root = path + '/' + folder_name
        if not os.path.isdir(folder_root):
            continue
        #for img_name in os.listdir(folder_root):
        img_name = os.listdir(folder_root)[0]    
        img_path = folder_root + '/' + img_name
        #读取图片
        sate_view = Image.open(img_path)          
        sate_view = sate_view.convert('RGB')
        if fi%2 == 0:
            sate_tensor = transform1(sate_view)
            folder_one = folder_name
            continue
        else :
            sate_tensor = torch.cat( (sate_tensor.unsqueeze(0),transform1(sate_view).unsqueeze(0)),0 )     
           
        print("processing: %d/%d"%(fi,total))
        with torch.no_grad():
            if view=='ground':
                sr,gr,dr = model(ground = sate_tensor)
            elif view=='satellite':
                sr,gr,dr = model(satellite = sate_tensor)
            elif view=='drone':
                sr,gr,dr = model(drone = sate_tensor)
                
        #print(len(result))
        #if isinstance(model, three_view_net):
            # result:  avg,dis,self.getSamples(avg,dis)
            # treat dis as a possibility?
            
            # or cal to a new one
            #samples = result[2]
            #tmp = result[0]
            #for i,item in enumerate(samples,0):
            #    tmp+=item
            #result = tmp/i
        if view=='ground':
            result = gr[0]
        elif view=='satellite':
            result = sr[0]
        elif view=='drone':
            result = dr[0]
        #    result = [0]
        label.append(folder_one)
        label.append(folder_name)
        datasets.append(result[0].numpy())
        datasets.append(result[1].numpy())
            
    return np.array(label),np.array(datasets)
            
def getdatasets(opt):
    transform1 = transforms.Compose([
        transforms.CenterCrop((opt.h, opt.w)), # 只能对PIL图片进行裁剪
        transforms.Pad( opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        )
        
    if opt.erasing_p>0:
        transform1 = transform1 +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:
        transform1 = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform1
        transform1 = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform1

    if opt.DA:
        transform1 = [ImageNetPolicy()] + transform1

    datasets = []
    
    foldeList =os.listdir(opt.data_dir + target_drone)
    random.shuffle(foldeList)
    for fi,folder_name in enumerate(foldeList,0):
        if fi% 10 == 0 :
            print('………………reading………………:%d/%d'%(fi,len(os.listdir(opt.data_dir + target_drone))))
    
        if fi % 2 == 0:
            anfolder_name = folder_name
            anfolder_root = opt.data_dir + target_drone + '/' + anfolder_name
            continue
        folder_root = opt.data_dir + target_drone + '/' + folder_name
        if not os.path.isdir(folder_root):
            continue
        
        #satelite item count
        satlist = os.listdir(opt.data_dir + target_satellite + '/' + folder_name)
        slen = len(satlist)
        #gound item count
        grolist = os.listdir(opt.data_dir + target_ground + '/' + folder_name)
        glen = len(grolist)
            
        
        #another satelite item count
        ansatlist = os.listdir(opt.data_dir + target_satellite + '/' + anfolder_name)
        anslen = len(ansatlist)
        #another gound item count
        angrolist = os.listdir(opt.data_dir + target_ground + '/' + anfolder_name)
        anglen = len(angrolist)
        
        andronelist =  os.listdir(anfolder_root)
        for i in [53]:
            img_name = os.listdir(folder_root)[i]
            
            #读取drone图片
            drone_view = Image.open(folder_root + '/' + img_name)          
            drone_view = drone_view.convert('RGB')
            drone_tensor = transform1(drone_view)
            
            #get satellite pic
            satellite_view = Image.open(opt.data_dir + target_satellite + '/' + folder_name+ '/' + satlist[i % slen])          
            satellite_view = satellite_view.convert('RGB')
            satellite_tensor = transform1(satellite_view)
          
            #get ground pic
            ground_view = Image.open(opt.data_dir + target_ground + '/' + folder_name + '/' + grolist[i % glen])          
            ground_view = ground_view.convert('RGB')
            ground_tensor = transform1(ground_view)
            
            #show_data.show( drone_tensor.unsqueeze(0), satellite_tensor.unsqueeze(0), ground_tensor.unsqueeze(0), ground_tensor.unsqueeze(0),'t1.jpg') 
            
            #an 读取drone图片
            androne_view = Image.open(anfolder_root + '/' + andronelist[i])          
            androne_view = androne_view.convert('RGB')
            androne_tensor = transform1(androne_view)
            
            #an get satellite pic
            ansatellite_view = Image.open(opt.data_dir + target_satellite + '/' + anfolder_name+ '/' + ansatlist[i % anslen])          
            ansatellite_view = ansatellite_view.convert('RGB')
            ansatellite_tensor = transform1(ansatellite_view)
          
            #an get ground pic
            anground_view = Image.open(opt.data_dir + target_ground + '/' + anfolder_name + '/' + angrolist[i % anglen])          
            anground_view = anground_view.convert('RGB')
            anground_tensor = transform1(anground_view)
            
            #show_data.show( androne_tensor.unsqueeze(0), ansatellite_tensor.unsqueeze(0), anground_tensor.unsqueeze(0), anground_tensor.unsqueeze(0),'t2.jpg') 
            
            item = (ground_tensor,drone_tensor,satellite_tensor,anground_tensor ,androne_tensor,ansatellite_tensor)   
            datasets.append(item)
    #torch.save(datasets,"dataTemp/Datasets200")
    return datasets
            
            
            
            
            
            
            
            
            