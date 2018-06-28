#-*- coding: utf-8 -*-

import cv2
import numpy as np

im = cv2.imread("./test_data/bird.jpg")
cv2.imshow("image_data",im)
cv2.waitKey(0) #これを入れないと画像が表示されずに終わる
#上記の場合=>何かキー入力があるまで待ち続け,キー入力があると次の処理に移行する

cv2.destroyAllWindows() #現在表示中のGUIの画像表示窓をすべて破壊する
