#-*- coding:utf-8 -*-
import cv2
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    #何かキーが押されるまで1ms待つ=>0の場合無限秒待つことになり更新されない
    k = cv2.waitKey(1)

    #キーが27=escの場合whileを抜ける
    if k == 27: #
        break

cap.release()
cv2.destroyAllWindows()
