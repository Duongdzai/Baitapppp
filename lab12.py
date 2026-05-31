import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('img/anhho.jpg', cv.IMREAD_GRAYSCALE)
img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f"Tìm thấy tổng cộng: {len(contours)} đường viền!")
cv.drawContours(img_color, contours, -1, (0, 0, 255), 2)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('1. Anh Nhi Phan Goc')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
plt.title('2. Da Ve Duong Vien (Do)')
plt.show()