# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:59:53 2021

@author: Jinda
"""



import read as rd
from tripResNet import ResNet
from torch.utils.data import TensorDataset as Dataset
from torch.utils.data import DataLoader, Sampler
import torchvision  as tv
import torch.optim as optim
import torch.nn as nn
import torch 
import numpy as np


import faiss 

mask = 100

#train_images = rd.load_train_images()
#torch.save(train_images, "tirpTrainImg")
train_images = torch.load("tirpTrainImg")
train_images = train_images[torch.arange(train_images.size(0))<mask]

#train_labels = rd.load_train_labels()
#torch.save(train_labels, "tirpTrainLb")
train_labels = torch.load("tirpTrainLb")
train_labels = train_labels [torch.arange(train_labels.size(0))<mask]

#test_images = rd.load_test_images()
#torch.save(test_images, "tirpTestImg")
test_images = torch.load("tirpTestImg")
test_images = test_images[torch.arange(test_images.size(0))<mask]

#test_labels = rd.load_test_labels()
#torch.save(test_labels, "tirpTestLb")
test_labels = torch.load("tirpTestLb")
test_labels = test_labels[torch.arange(test_labels.size(0))<mask]

device = torch.device("cpu")

#resnet = torch.load('ResNet.par')
resnet = torch.load('DisTriNetWithTwo.par')

print(train_images.size())
print('cal orig train DataSet')
train_search = []
train_distri = []
with torch.no_grad():
    for i,data in enumerate(train_images):
        result = resnet(data.unsqueeze(0))
        train_search.append(result[0].reshape(-1).numpy())
        train_distri.append(result[1].numpy())
    
train_search = np.array(train_search)
train_distri = np.array(train_distri)
#out = (长度为numclasss的均值向量，方差向量，【N*samples向量】) 
    
print('cal orig test DataSet')
test_search = []
test_distri = []
with torch.no_grad():
    for i,data in enumerate(test_images):
        result = resnet(data.unsqueeze(0))
        test_search.append(result[0].reshape(-1).numpy())
        test_distri.append(result[1].numpy())
        
test_search = np.array(test_search)
test_distri =  np.array(test_distri)
#search test in train

index = faiss.IndexFlatL2(len(train_search[0]))    
#10 dimension vector
index.add(train_search)

D, I = index.search(test_search, 10)

#test_search = m * (长度为numclasss的均值向量，方差向量，【N*samples向量】) 
test_meanDistribute = []
#D = np.mean(D,axis=1)
for i in range(len(test_distri)):
    test_meanDistribute.append(np.sum(a=test_distri[i],axis = 1))


#I[0] I[1] ... I[10]  the index of originImag
#其中example_batch[0] 维度为torch.Size([8, 1, 100, 100])

#concatenated = train_images[I[0][0]].squeeze(0)
#for n in range(1,10):
#    concatenated = torch.cat((concatenated,train_images[I[0][n]].squeeze(0)),1)
#print(concatenated.size())
    
#创建新的figure
#plt.figure()
#plt.imshow(concatenated)
#plt.show()
#plt.pause(10)
#plt.close()

acc = 0.0
newAc = 0.0

for i,result in enumerate(I):
    #distance and index result
    index = result
    testLabel = test_labels[i].item();
    count = 0
    loss = D[i]
    rresult = []
    print('***************************************************************************')
    print('----test_distribution----')
    print(testLabel)
    print(test_distri[i])
    print('---- label,distibute ----')
    for j in index:
        print(train_labels[j])
        print(train_distri[j])
        #rresult.append(train_labels[j].item())
        #rresult.append(test_meanDistribute[i])
        if testLabel == train_labels[j].item() :
            count+=1
    newAc = count / 10.0
    acc = (i*acc + newAc)/(i+1)
    #if i %100 == 0:
    
    #print('testLbl:%d with [label,distribute,~] '%testLabel,rresult,'| Loss',loss,'| Acc%0.3f'%(acc))
   
print('%d/100 Total Loss%0.3f | Acc%0.3f'%(i,loss,acc))
        
        