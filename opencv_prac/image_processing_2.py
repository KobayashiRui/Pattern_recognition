#-*- coding:utf-8 -*-
"""
エッジ検出を行う
"""
import cv2

img = cv2.imread('./test_data/bird.jpg')
hight = img.shape[0]
width = img.shape[1]
img = cv2.resize(img,(width/4,hight/4))

#グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Cannyアルゴリズムにてエッジ検出
canny_edges = cv2.Canny(img_gray,220,400)
#引数1:最小閾値(弱edge) 引数2:最大閾値(強edge)
#強edge: 画素変化が著しい=エッジの可能性を持つ箇所を検出するのに用いられる
#=>エッジ検出への影響大
#弱edge: 検出されたエッジが繋がっているか否かを判定するのに用いられる
#=>強edgeで検出されたエッジがどこまで続いているかに影響大

cv2.imshow("origin_image",img)
cv2.imshow("gray_image",img_gray)
cv2.imshow("Canny_edges",canny_edges)

cv2.waitKey(0)

cv2.destroyAllWindows()
