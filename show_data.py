import sys
import torch
import os
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage
#target_root = 'data/train/drone'
#target_root = 'data/train/street'
#target_root = 'data/train/satellite'
target_root = 'data/train/google'

def pad(inp, pad = 3):
    #print(inp.size)
    h, w = inp.size
    bg = np.zeros((h+2*pad, w+2*pad, len(inp.mode)))
    bg[pad:pad+h, pad:pad+w, :] = inp
    return bg

def showMid(d,path):
    
    npad = 3
    nrow = round(len(d) ** 0.5)
    white_col = np.ones((24,len(d[0])))*255
    white_row = np.ones(((len(d[0])+24) * nrow,24))*255
    
    for i,who in enumerate(d):
        img = ToPILImage()(who)
        if i==0 :
            result = np.concatenate((img,white_col), axis = 0)
        elif i % nrow == 0:
            if i==nrow:
                whole = result
                result = np.concatenate((img,white_col), axis = 0)
            else:
                whole = np.concatenate((whole,white_row,result), axis = 1)
                result =  np.concatenate((img,white_col), axis = 0)
        else:
            result = np.concatenate((result,white_col,img), axis = 0)
        
    if (i-1)%nrow !=0 :
        result = np.concatenate((result,np.ones((len(whole) - len(result),len(d[0])))*255), axis = 0)
        whole = np.concatenate((whole,white_row,result), axis = 1)
    final = Image.fromarray(whole.astype('uint8'))
    final.save(path + '.jpg')
    
    
    
def show(d1,d2,d3,d4,path):
    
    npad = 3
    inputs= d1
    inputs2 = d2
    inputs3 = d3
    inputs4 = d4
    for i in range(len(inputs)):
        
        img1 = ToPILImage()(inputs[i].squeeze(0))
        img2 = ToPILImage()(inputs2[i].squeeze(0))
        img3 = ToPILImage()(inputs3[i].squeeze(0))
        img4 = ToPILImage()(inputs4[i].squeeze(0))
         
        tmp1 = pad(img1, pad=npad)
        tmp2 = pad(img2, pad=npad)
        tmp3 = pad(img3, pad=npad)
        tmp4 = pad(img4, pad=npad)
        result2 = np.concatenate((tmp1,tmp2,tmp3,tmp4), axis=1)
        if i==0:
            result = result2
        else:
            result = np.concatenate((result,result2), axis = 0)
        
    tmp1 = Image.fromarray(result.astype('uint8'))
    tmp1.save(path)
        
        
if __name__ == '__main__':
        
    count = 0
    ncol = 20
    nrow = 25
    npad = 3
    im = {}
    
    white_col = np.ones((128+2*npad,24,3))*255
    
    for folder_name in os.listdir(target_root):
        folder_root = target_root + '/' + folder_name
        if not os.path.isdir(folder_root):
            continue
        for img_name in os.listdir(folder_root):
            input1 = Image.open(folder_root + '/' + img_name)
            input1 = input1.convert('RGB')
            print(folder_root + '/' + img_name)
            input1 = input1.resize( (128, 128))
            # Start testing
            tmp = pad(input1, pad=npad)
            if count%ncol == 0:
                im[count//ncol] = tmp
            else:
                im[count//ncol] = np.concatenate((im[count//ncol], white_col, tmp), axis=1)
            count +=1
            if 'drone' in target_root:
                break
        if count > nrow*ncol:
            break
            
    
    first_row = np.ones((128+2*npad,128+2*npad,3))*255
    white_row = np.ones( (24,im[0].shape[1],3))*255
    for i in range(nrow):
        if i == 0:
            pic = im[0]
        else:
            pic = np.concatenate((pic, im[i]), axis=0)
        pic = np.concatenate((pic, white_row), axis=0)
        #first_row = np.concatenate((first_row, white_col, im[i][0:256+2*npad, 0:256+2*npad, 0:3]), axis=1)
    
    #pic = np.concatenate((first_row, white_row, pic), axis=0)
    pic = Image.fromarray(pic.astype('uint8'))
    pic.save('sample_%s.jpg'%os.path.basename(target_root))
    #pic.save('sample.jpg')
