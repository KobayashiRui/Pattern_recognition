#-*- coding: utf-8 -*-

import cv2
import numpy as np

image_data = cv2.imread("./test_data/bird.jpg")

#元サイズだとでかいので1/4にしとく
hight = image_data.shape[0] #高さ
width = image_data.shape[1] #幅
image_data = cv2.resize(image_data,(width/4,hight/4)) 

#下記の作業がRGBの値を分離してその値を白黒画像で表す
RGB = cv2.split(image_data)
Blue = RGB[0]#Blueを0~255で表す=>0~255のみなので白黒
Green = RGB[1]
Red = RGB[2]

#単色画像を作成する
zeros = np.zeros((hight/4, width/4), image_data.dtype)
img_blue = cv2.merge((Blue, zeros,zeros))
img_green = cv2.merge((zeros,Green,zeros))
img_red = cv2.merge((zeros,zeros,Red))

cv2.imshow("Blue",img_blue)
cv2.imshow("Green",img_green)
cv2.imshow("Red",img_red)
cv2.imshow("Add",img_blue+img_green+img_red) #RGBの単色データ同士を足すと元データになる

cv2.waitKey(0)
cv2.destroyAllWindows()
