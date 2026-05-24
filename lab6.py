import sys
import cv2
import numpy as np
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
img2 = img1.copy() 
cv2.imshow("Anh 1 (Query)", img1)
cv2.imshow("Anh 2 (Database)", img2)
cv2.waitKey(0) 
cv2.destroyAllWindows()
print("BÀI TOÁN IMAGE RETRIEVAL (TRUY XUẤT ẢNH)\n")
print("1. Phương pháp Raw Data:")
raw1 = cv2.resize(img1, (64, 64)).astype(float)
raw2 = cv2.resize(img2, (64, 64)).astype(float)
mse = np.mean((raw1 - raw2) ** 2)
print(f" -> Độ chênh lệch pixel (MSE): {mse} (Kết quả = 0 nghĩa là giống hệt nhau)\n")
print("2. Phương pháp Histogram:")
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
cv2.normalize(hist1, hist1)
cv2.normalize(hist2, hist2)
score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
print(f" -> Độ tương đồng màu sắc (Correlation): {score} (Kết quả = 1.0 nghĩa là giống hệt nhau)")
