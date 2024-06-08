import cv2
import os
import numpy as np
from .data_preparing import preprocessed_images, ROOT_PATH

input_folder = preprocessed_images
output_folder = os.path.join(ROOT_PATH, 'same_statio')
target_size = (300, 300)  # Kích thước mong muốn của vật thể


# Lặp qua tất cả các tệp tin trong thư mục đầu vào
def same_statio_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ xử lý các tệp tin hình ảnh
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Đọc hình ảnh
            img = cv2.imread(image_path)

            # Chỉnh sửa kích thước vật thể
            resized_object = cv2.resize(img, target_size)

            # Khởi tạo khung hình mới có kích thước 500x500
            frame = np.zeros((500, 500, 3), dtype=np.uint8)

            # Tính toán tọa độ để đặt vật thể vào giữa khung hình
            x = int((500 - target_size[0]) / 2)
            y = int((500 - target_size[1]) / 2)

            # Tính toán tọa độ của vật thể trong khung hình
            obj_x = x
            obj_y = y
            obj_width = target_size[0]
            obj_height = target_size[1]

            # Đặt vật thể vào khung hình
            frame[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width] = resized_object

            # Lưu hình ảnh đã chỉnh sửa vào thư mục đầu ra
            cv2.imwrite(output_path, frame)


# same_statio_images(input_folder)
# python3 data_process_preparing/same_statio.py
