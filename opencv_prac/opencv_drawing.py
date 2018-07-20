#-*- coding:utf-8 -*-
import cv2
import numpy as np

img = np.full((210,425,3),128,dtype=np.uint8)

cv2.line(img, (50,10),(125,60),(255,0,0))
cv2.line(img, (50,60),(125,10),(0,255,0),thickness=4,lineType=cv2.LINE_AA)


cv2.imshow("test_drawing",img)
cv2.waitKey(0)

cv2.destroyAllWindows()
