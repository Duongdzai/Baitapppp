import cv2
import random
image_path = 'img/anhho.jpg'
img = cv2.imread(image_path)
if img is None:
    print("Lỗi: Không tìm thấy file ảnh", image_path)
else:
    import random
    random_sang = random.randint(0, 255)
    random_toi = random.randint(0, 255)
    random_tuong_phan = random.uniform(1.0, 5.0) 
    print(f"-> Tăng sáng ngẫu nhiên: +{random_sang}")
    print(f"-> Giảm sáng ngẫu nhiên: -{random_toi}")
    print(f"-> Tương phản ngẫu nhiên: {random_tuong_phan:.1f}")
    img_sang = cv2.convertScaleAbs(img, beta=random_sang)       
    img_toi = cv2.convertScaleAbs(img, beta=-random_toi)        
    img_tuong_phan = cv2.convertScaleAbs(img, alpha=random_tuong_phan) 
    cv2.imshow('1. Anh Goc', img)
    cv2.imshow('2. Anh Sang', img_sang)
    cv2.imshow('3. Anh Toi', img_toi)
    cv2.imshow('4. Anh Tuong Phan', img_tuong_phan)
    print("Nhấn phím bất kỳ trên ảnh để thoát")
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
