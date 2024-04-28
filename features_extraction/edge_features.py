import os
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
import cv2
import matplotlib.pyplot as plt
import csv

image_folder = 'D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images'

output_csv_file = 'edges_result.csv'


def solve_edge_feature(image_path):
    image = imread(image_path,as_gray=True)

    #calculating horizontal edges using prewitt kernel
    edges_prewitt_horizontal = prewitt_h(image)

    #calculating vertical edges using prewitt kernel
    edges_prewitt_vertical = prewitt_v(image)

    # Chuyển đổi ma trận thành danh sách 1D
    edges_list = edges_prewitt_vertical.flatten()

     # Tạo danh sách tên cột
    column_names = ['Image']
    column_names.extend(['Edge{}'.format(i+1) for i in range(len(edges_list))])

    # Ghi tên ảnh và danh sách cạnh vào tệp CSV
    with open(output_csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(output_csv_file).st_size == 0:  # Nếu tệp CSV rỗng, ghi tên cột
            writer.writerow(column_names)
        writer.writerow([os.path.basename(image_path), *edges_list])

# Duyệt qua tất cả các thư mục con trong thư mục process_image
for root, dirs, files in os.walk(image_folder):
    # Duyệt qua tất cả các tệp trong thư mục hiện tại
    for file in files:
         # Xác định đường dẫn tới tệp ảnh
        image_path = os.path.join(root, file)
        
        # Gọi hàm solve_edge_feature() trên tệp ảnh
        solve_edge_feature(image_path)