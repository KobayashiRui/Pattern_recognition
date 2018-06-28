#-*- coding:utf-8 -*-
#取得する画像のサイズを変換 resizeの利用
import cv2
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    height = frame.shape[0] #高さ
    width = frame.shape[1] #幅
    channel = frame.shape[2] #画像のチャンネル数 RGBで3

    frame = cv2.resize(frame,(width/2,height/2))
    #引数1:サイズを変更する画像
    #引数2:変更後のサイズのタプル=> タプル内は(幅,高さ)

    cv2.imshow('frame',frame)

    #何かキーが押されるまで1ms待つ=>0の場合無限秒待つことになり更新されない
    k = cv2.waitKey(1)

    #キーが27=escの場合whileを抜ける
    if k == 27: #
        break

cap.release()
cv2.destroyAllWindows()
