import cv2
import numpy as np
import os

def tu_lam_sobel(image, kernel):
    """Hàm tự làm để tính Gradient bằng Tích chập (giống nguyên lý của OpenCV)"""
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    
    # LƯU Ý QUAN TRỌNG: Phải dùng float32 vì đạo hàm có thể ra số âm 
    # (khi màu đổi từ điểm sáng sang điểm tối)
    output = np.zeros_like(image, dtype=np.float32)
    
    # Quét kernel qua từng pixel (bỏ qua viền)
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            vung_anh = image[y-pad : y+pad+1, x-pad : x+pad+1]
            output[y, x] = np.sum(vung_anh * kernel)
            
    # Lấy TRỊ TUYỆT ĐỐI của các số âm, rồi ép về uint8 (0-255) để thành ảnh chuẩn
    return np.uint8(np.absolute(output))

def main():
    # 1. Đọc ảnh xám
    thu_muc_hien_tai = os.path.dirname(os.path.abspath(__file__))
    duong_dan_anh = os.path.join(thu_muc_hien_tai, 'img', 'anhho.jpg')
    img = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Lỗi: Không tìm thấy file tại '{duong_dan_anh}'")
        return
    img = cv2.resize(img, (350, 350))

    # Khai báo ma trận (Kernel) Sobel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32)
    
    # Chạy hàm tự làm
    tu_lam_sobelX = tu_lam_sobel(img, kernel_x)
    tu_lam_sobelY = tu_lam_sobel(img, kernel_y)
    tu_lam_tong = cv2.bitwise_or(tu_lam_sobelX, tu_lam_sobelY)
    cv2_sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    cv2_sobelX = np.uint8(np.absolute(cv2_sobelX))
    
    cv2_sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    cv2_sobelY = np.uint8(np.absolute(cv2_sobelY))
    
    cv2_sobel_tong = cv2.bitwise_or(cv2_sobelX, cv2_sobelY)
    
    # Laplacian (Phát hiện cạnh mọi hướng)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    cv2.imshow("0. Anh Goc", img)
    
    # So sánh Sobel X (Phát hiện cạnh dọc)
    cv2.imshow("1A. Tu Lam (Sobel X)", tu_lam_sobelX)
    cv2.imshow("1B. OpenCV (Sobel X)", cv2_sobelX)
    
    # So sánh Sobel Tổng (Cả dọc + ngang)
    cv2.imshow("2A. Tu Lam (Sobel Tong)", tu_lam_tong)
    cv2.imshow("2B. OpenCV (Sobel Tong)", cv2_sobel_tong)
    cv2.imshow("3. OpenCV (Laplacian)", laplacian)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
