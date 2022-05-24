import cv2
import numpy as np
import sys

default_file = './calibrating/testcolors.png'
filename = sys.argv[1] if len(sys.argv) > 1 else default_file

imageRGB = cv2.imread(filename)
imageHSV = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2HSV)

print(imageHSV[125, 637])
cv2.imshow("Res", imageHSV)
# cv2.setMouseCallback('mouseRGB', mouseRGB)
cv2.waitKey(0)


# def mouseRGB(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
#         colorsB = image[y,x,0]
#         colorsG = image[y,x,1]
#         colorsR = image[y,x,2]
#         colors = image[y,x]
#         print("Red: ",colorsR)
#         print("Green: ",colorsG)
#         print("Blue: ",colorsB)
#         print("BRG Format: ",colors)
#         print("Coordinates of pixel: X: ",x,"Y: ",y)