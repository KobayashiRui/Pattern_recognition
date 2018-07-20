#-*- coding:utf-8 -*-
"""
2値化を行う
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./line_trace_imagedatat/WIN_20180720_07_51_54_Pro.jpg')
hight = img.shape[0]
width = img.shape[1]
img = cv2.resize(img,(width/2,hight/2))
print("{}:: {}".format(hight/2,width/2))

#グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = img_gray[200:260,0:640]

cv2.imshow("gray_image",img_gray)

cv2.waitKey(0)

cv2.destroyAllWindows()

histr = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(histr,color = 'b')
plt.show()

#二値化
thresh=160 #pixelが100より大きいと1=白とする
max_pixel = 255
ret, img_dst = cv2.threshold(img_gray,thresh,max_pixel, cv2.THRESH_BINARY)
img_binary = img_dst / 255
line_data = np.sum(img_binary, axis=0)
line_data_left = line_data[0:((line_data.shape[0]/2)-1)]
print(line_data_left.shape)
line_data_right = line_data[(line_data.shape[0]/2):(line_data.shape[0]-1)]
print(line_data_right.shape)
print(line_data_left.sum())
print(line_data_right.sum())

#line_data = line_data.reshape(-1,1)
print(line_data.shape)
x_data = range(-line_data.shape[0]/2,line_data.shape[0]/2)
plt.plot(x_data,line_data)
plt.show()


cv2.imshow("binarization_image",img_dst)

cv2.waitKey(0)

cv2.destroyAllWindows()
