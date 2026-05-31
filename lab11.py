import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh dạng ảnh xám (Sửa lại đường dẫn chuẩn của bạn)
img = cv.imread('img/anhho.jpg', cv.IMREAD_GRAYSCALE)

# Nếu không có ảnh thật, tự tạo 1 hình vuông trắng trên nền đen để học thử
if img is None:
    img = np.zeros((200, 200), dtype=np.uint8)
    cv.rectangle(img, (50, 50), (150, 150), 255, -1) # Vẽ hình vuông trắng
else:
    # Chuyển ảnh thật thành ảnh nhị phân (Đen / Trắng)
    _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# 2. Tạo một bộ lọc (Kernel) kích thước 5x5 toàn số 1
kernel = np.ones((5, 5), np.uint8)

# 3. Áp dụng 2 phép toán cơ bản nhất
erosion = cv.erode(img, kernel, iterations=1)   # Phép bào mòn (làm mảnh nét)
dilation = cv.dilate(img, kernel, iterations=1) # Phép giãn nở (làm dày nét)

# 4. Hiển thị kết quả ra màn hình để đối chiếu
titles = ['Anh Goc', 'Bao Mon (Erode)', 'Gian No (Dilate)']
images = [img, erosion, dilation]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([]) # Ẩn trục tọa độ cho dễ nhìn

plt.show()