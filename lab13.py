import cv2 
import numpy as np

img = cv2.imread('img/hinhhoc.jpg')
img = cv2.resize(img, (600, 600))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(cnt)
        if len(approx) == 3:
                ten_hinh = "Hinh Tam Giac"
        elif len(approx) == 4:    
            # 1. Tính độ dài cạnh thứ nhất (nối từ đỉnh 0 đến đỉnh 1)
            p1 = approx[0][0]
            p2 = approx[1][0]
            canh_1 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            p3 = approx[2][0]
            canh_2 = np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
            ratio = min(canh_1, canh_2) / max(canh_1, canh_2)
            if ratio > 0.85:
                ten_hinh = "Hinh Vuong"
            else:
                ten_hinh = "Hinh Chu Nhat"  
        cv2.putText(img, ten_hinh, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow('Tự nhận dạng hình',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 

