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
img = cv2.resize(img,(width/4,hight/4))

#グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

histr = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(histr,color = 'k')
plt.show()

#二値化
thresh=160 #pixelが100より大きいと1=白とする
max_pixel = 255
ret, img_dst = cv2.threshold(img_gray,thresh,max_pixel, cv2.THRESH_BINARY)

#cv2.imshow("gray_image",img_gray)
cv2.imshow("binarization_image",img_dst)

cv2.waitKey(0)

cv2.destroyAllWindows()
