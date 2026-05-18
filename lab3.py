import cv2
import numpy as np

# --- CÔNG CỤ TÌM DẢI MÀU ---
def get_hsv_bounds(b, g, r):
    """Chuyển đổi màu BGR sang HSV và tính khoảng màu cần lọc"""
    color = np.uint8([[[b, g, r]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h = hsv_color[0][0][0]
    
    # Khoảng màu (giảm 10 và tăng 10 từ giá trị H)
    lower_bound = np.array([max(0, h - 10), 100, 100])
    upper_bound = np.array([min(179, h + 10), 255, 255])
    return lower_bound, upper_bound

# 1. Định nghĩa màu cần lọc (Ví dụ: Xanh dương B=255, G=0, R=0)
lower_color, upper_color = get_hsv_bounds(255, 0, 0)
print(f"Đang lọc màu - Ngưỡng dưới: {lower_color}, Ngưỡng trên: {upper_color}")

# 2. Mở camera
cap = cv2.VideoCapture(0)

# Khởi tạo cửa sổ
cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Result', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Chuyển ảnh sang hệ màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 4. Tạo mặt nạ lọc màu
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 5. Cắt phần màu từ ảnh gốc
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # 6. Hiển thị 3 cửa sổ
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)

    # Bấm 'q' hoặc phím ESC để thoát
    if cv2.waitKey(30) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
