#-*- coding:utf-8 -*-
"""
2値化を行う
"""
import cv2

img = cv2.imread('./test_data/bird.jpg')
hight = img.shape[0]
width = img.shape[1]
img = cv2.resize(img,(width/4,hight/4))

#グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#二値化
thresh=150 #pixelが100より大きいと1=白とする
max_pixel = 255
ret, img_dst = cv2.threshold(img_gray,thresh,max_pixel, cv2.THRESH_BINARY)

cv2.imshow("gray_image",img_gray)
cv2.imshow("binarization_image",img_dst)

cv2.waitKey(0)

cv2.destroyAllWindows()
