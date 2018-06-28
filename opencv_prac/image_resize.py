#-*- coding: utf-8 -*-

import cv2
import numpy as np

image_data = cv2.imread("./test_data/bird.jpg")

hight = image_data.shape[0] #高さ
width = image_data.shape[1] #幅

image_data = cv2.resize(image_data,(width/2,hight/2)) #画像サイズを半分にする
#幅と高さの設定に注意

#resizeの値が割り切れないと落ちるらしい
#=> image_data = cv2.resize(image_data,(round(width/2,round(hight/2))))
#上記のようにroundを使って丸めるといいらしい

cv2.imshow("image_data",image_data)
cv2.waitKey(0) #これを入れないと画像が表示されずに終わる
#上記の場合=>何かキー入力があるまで待ち続け,キー入力があると次の処理に移行する

cv2.destroyAllWindows() #現在表示中のGUIの画像表示窓をすべて破壊する
