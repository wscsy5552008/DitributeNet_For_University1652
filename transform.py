import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from PIL import Image
import math

target_satellite = '../data/train/satellite'
target_test_satellite = '../data/test/gallery_satellite'
polar_test_satellite = '../data/test/gallery_polar_satellite'
polar_satellite = '../data/train/polar_satellite'


target_drone = '../data/train/drone'
target_test_drone = '../data/test/gallery_drone'
polar_drone = '../data/train/polar_drone'
polar_test_drone = '../data/test/gallery_polar_drone'

class Conversion:
    def __init__(self,T_x ,T_y ,x_0 ,y_0 ,d_thita ,d_r,):
        self.img = None
        self.T_x = T_x
        self.T_y = T_y
        self.x_0 = x_0
        self.y_0 = y_0
        self.d_thita = d_thita
        self.d_r = d_r
        self.PI = np.pi
        self.dst = np.zeros((T_y, T_x,3), dtype=float)

    #双线性
    def interpolate_bilinear(self,xi, xf, xc, yi, yf, yc):
        if yf == yc & xc == xf:
            out = self.img[xc][yc]
        elif yf == yc:
            out = (xi - xf) * self.img[xc][yf] + (xc - xi) * self.img[xf][yf]
        elif xf == xc:
            out = (yi - yf) * self.img[xf][yc] + (yc - yi) * self.img[xf][yf]
        else:
            inter_r1 = (xi - xf) * self.img[xc][yf] + (xc - xi) * self.img[xf][yf]
            inter_r2 = (xi - xf) * self.img[xc][yc] + (xc - xi) * self.img[xf][yc]
            out = (yi - yf) * inter_r2 + (yc - yi) * inter_r1
        return out

    def cal(self,img):
        self.img = img
        self.dst = np.zeros((T_y, T_x,3), dtype=float)
        for thita in range(self.T_x):
            for r in range(self.T_y):
                if r == 0:
                    self.dst[r][thita] = self.img[self.x_0][self.y_0]
                else:
                    self.trans(thita,r)
        return self.dst
    def trans(self,thita,r):
        
        xi = self.x_0 + r * self.d_r * np.cos(thita * self.d_thita / 180 * self.PI)-1
        xf = np.int_(np.floor(xi))
        xc = np.int_(np.ceil(xi))
        yi = self.y_0 + r * self.d_r * np.sin(thita * self.d_thita / 180 * self.PI)-1
        yf = np.int_(np.floor(yi))
        yc = np.int_(np.ceil(yi))
        #self.dst[i][y] = self.interpolate_adjacent(xi,xf,xc,yi,yf,yc)
        self.dst[r][thita] = self.interpolate_bilinear( xi, xf, xc,yi, yf, yc)
        

#读取图片
H = 384
W = 384

T_x = np.int_(np.floor(W * 4))
T_y = np.int_(np.floor(H))

x_0 = np.int_(np.floor(W/2))
y_0 = np.int_(np.floor(H/2))

d_thita = 360/T_x #划分圆周
d_r = W/ (2 * T_y) #划分半径

result = np.zeros((T_x, T_y, 3), dtype=float)
img  = Conversion(T_x ,T_y ,x_0 ,y_0 ,d_thita ,d_r)

#for i in range(3):
#    img  = Conversion(T_x ,T_y ,x_0 ,y_0 ,d_thita ,d_r,sate_view[:,:,i])
#    tmp = img.cal()
#    for h in range(T_y):
#        for w in range(T_x):
#            result[w][h][i] = tmp[w][h]

foldeList =os.listdir(target_drone)
if not os.path.isdir(polar_drone):
    os.mkdir(polar_drone)
for folder_name in foldeList:
    fileName = os.listdir( target_drone + '/' +folder_name)[0]
    img_path = target_drone + '/' + folder_name + '/' + fileName
    print('processing: ' + folder_name + '/' + fileName)
    sate_view = Image.open(img_path)          
    sate_view = sate_view.convert('RGB')
    sate_view = sate_view.resize((H,W))
    sate_view = np.asarray(sate_view)
    result = img.cal(sate_view)
    result = Image.fromarray(np.uint8(result))
    
    if not os.path.isdir(polar_drone  + '/' + folder_name):
        os.mkdir(polar_drone  + '/' + folder_name)
    result.save(polar_drone  + '/' + folder_name + '/' + 'polar.jpg')
    break