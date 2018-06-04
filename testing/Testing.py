import numpy as np
import cv2

img = cv2.imread('test1-color.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
depth = cv2.imread('test2-depth.jpg')

mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

cv2.imshow('original', img)
cv2.imshow('grayscale', gray)
#cv2.imshow('processed', mask)
cv2.imshow('depth', depth)

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
