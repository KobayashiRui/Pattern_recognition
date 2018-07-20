#-*- coding:utf-8 -*-
import cv2

def Print_webcam_setting(web_cam):
    print(web_cam.get(cv2.CAP_PROP_FPS)) #FPSの設定を表示
    print(web_cam.get(cv2.CAP_PROP_FRAME_WIDTH)) #カメラの幅の設定の表示
    print(web_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))#カメラの高さの設定を表示

def Set_webcam_setting(web_cam):
    web_cam.set(cv2.CAP_PROP_FPS, 30) #FPSの設定
    web_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320) #幅の設定
    web_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) #高さの設定

cap = cv2.VideoCapture(0)

Print_webcam_setting(cap)
Set_webcam_setting(cap)
Print_webcam_setting(cap)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    k = cv2.waitKey(1)

    if k == 27: #
        break

cap.release()
cv2.destroyAllWindows()
