import cv2
import numpy as np
import os

def tu_viet_filter2D(image, kernel):
    """Hàm tự viết Tích chập (chỉ xử lý ảnh Xám để code ngắn và dễ hiểu nhất)"""
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    
    # Tạo ảnh kết quả chứa toàn số 0
    output = np.zeros_like(image, dtype=np.float32)
    
    # Quét kernel qua từng pixel của ảnh (bỏ qua viền ngoài cùng cho đơn giản)
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            # Cắt ra một vùng ảnh nhỏ bằng kích thước kernel
            vung_anh = image[y-pad : y+pad+1, x-pad : x+pad+1]
            
            # Nhân ma trận vùng ảnh với kernel, rồi cộng tất cả lại (Tích chập)
            output[y, x] = np.sum(vung_anh * kernel)
            
    # Đưa các giá trị về chuẩn [0, 255]
    return np.clip(output, 0, 255).astype(np.uint8)

def main():
    # 1. Đọc ảnh 'anhho.jpg' (Tính toán đường dẫn tự động để không bị lỗi Not Found)
    thu_muc_hien_tai = os.path.dirname(os.path.abspath(__file__))
    duong_dan_anh = os.path.join(thu_muc_hien_tai, 'img', 'anhho.jpg')
    
    img = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Lỗi: Không tìm thấy file tại '{duong_dan_anh}'")
        return
        
    # Resize lại ảnh cho nhỏ bớt để dễ nhìn trên màn hình
    img = cv2.resize(img, (300, 300))
    
    # 2. Tạo Kernel làm mờ 5x5
    kernel = np.ones((5, 5), np.float32) / 25.0
    
    # 3. DÙNG HÀM TỰ VIẾT
    anh_tu_viet = tu_viet_filter2D(img, kernel)
    
    # 4. DÙNG THƯ VIỆN OPENCV
    anh_filter2D = cv2.filter2D(img, -1, kernel)      # 2D Convolution
    anh_blur = cv2.blur(img, (5, 5))                  # Averaging
    anh_gaussian = cv2.GaussianBlur(img, (5, 5), 0)   # Gaussian
    anh_median = cv2.medianBlur(img, 5)               # Median (Khử nhiễu hột cực tốt)
    
    # 5. Hiển thị so sánh
    cv2.imshow("1. Anh Goc", img)
    cv2.imshow("2. Ham Tu Viet", anh_tu_viet)
    cv2.imshow("3. OpenCV (Filter2D)", anh_filter2D)
    cv2.imshow("4. OpenCV (Median Blur)", anh_median)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
