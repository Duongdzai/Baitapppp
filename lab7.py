import cv2
import numpy as np

def create_sample_image():
    # Tạo một bức ảnh đen (dummy image) kích thước 300x300
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Vẽ một số hình khối để tạo các đường viền (edges)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1) # Hình vuông trắng
    cv2.circle(img, (200, 200), 50, (0, 0, 255), -1)              # Hình tròn đỏ
    cv2.line(img, (50, 250), (250, 50), (0, 255, 0), 5)           # Đường thẳng xanh lá
    
    return img

def main():
    print("Đang chạy chương trình demo tích chập (Convolution)...")
    
    # 1. Khởi tạo ảnh (Có thể thay bằng cv2.imread('duong_dan_anh.jpg'))
    image = create_sample_image()
    
    # 2. Định nghĩa các Kernels (Filters)
    
    # Kernel làm mờ (Average Blur 5x5)
    kernel_blur = np.ones((5, 5), np.float32) / 25.0
    
    # Kernel làm sắc nét (Sharpen)
    kernel_sharpen = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])
    
    # Kernel phát hiện biên (Sobel Y - Cạnh ngang)
    kernel_edge = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])

    # 3. Thực hiện phép tích chập bằng cv2.filter2D
    blurred_img = cv2.filter2D(image, -1, kernel_blur)
    sharpened_img = cv2.filter2D(image, -1, kernel_sharpen)
    edged_img = cv2.filter2D(image, -1, kernel_edge)

    # 4. Hiển thị kết quả
    cv2.imshow("1. Anh Goc (Original)", image)
    cv2.imshow("2. Lam mo (Blur)", blurred_img)
    cv2.imshow("3. Lam sac net (Sharpen)", sharpened_img)
    cv2.imshow("4. Phat hien bien (Edge Detection)", edged_img)

    print("Nhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
