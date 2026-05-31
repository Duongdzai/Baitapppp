import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread('anh.jpg', cv.IMREAD_GRAYSCALE)
if img is None:
  img = cv.imread('img/anhho.jpg', cv.IMREAD_GRAYSCALE)
_, simple = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
_, otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
titles = ['Ảnh gốc', 'Ngưỡng cố định', 'Thích ứng', 'Otsu']
images = [img, simple, adaptive, otsu]
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()