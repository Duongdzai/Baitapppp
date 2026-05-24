import cv2
import numpy as np
import matplotlib.pyplot as plt
def calculate_histogram_scratch(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(256, dtype=int)
    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            pixel_intensity = image[y, x]
            hist[pixel_intensity] += 1
    return hist
if __name__ == "__main__":
    image_path = 'img/anhho.jpg' 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}. Tạo ảnh ngẫu nhiên để thử nghiệm...")
        image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    my_hist = calculate_histogram_scratch(image)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title('Ảnh gốc')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.plot(my_hist, color='black')
    plt.bar(range(256), my_hist, color='gray', alpha=0.7)
    plt.title('Histogram (from scratch)')
    plt.xlabel('Cường độ pixel (0-255)')
    plt.ylabel('Số lượng pixel')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()
