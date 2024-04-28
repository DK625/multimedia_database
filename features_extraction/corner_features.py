import os
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
import cv2
import matplotlib.pyplot as plt
import csv

image_folder = 'D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images'

output_csv_file = 'corner_result.csv'

def solve_corner_featur(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using the Harris method
    dst = cv2.cornerHarris(gray, 3, 5, 0.1)
    
    # Create a boolean bitmap of corner positions
    corners = dst > 0.05 * dst.max()
    
    # Find the coordinates from the boolean bitmap
    coord = np.argwhere(corners)
    
    # Draw circles on the coordinates to mark the corners
    for y, x in coord:
        cv2.circle(img, (x,y), 3, (0,0,255), -1)

    # Chuyển đổi ma trận thành danh sách 1D
    corners_list = dst.flatten()

     # Tạo danh sách tên cột
    column_names = ['Image']
    column_names.extend(['Corner{}'.format(i+1) for i in range(len(corners_list))])

     # Ghi tên ảnh và danh sách cạnh vào tệp CSV
    with open(output_csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(output_csv_file).st_size == 0:  # Nếu tệp CSV rỗng, ghi tên cột
            writer.writerow(column_names)
        writer.writerow([os.path.basename(image_path), *corners_list])


# Duyệt qua tất cả các thư mục con trong thư mục process_image
for root, dirs, files in os.walk(image_folder):
    # Duyệt qua tất cả các tệp trong thư mục hiện tại
    for file in files:
         # Xác định đường dẫn tới tệp ảnh
        image_path = os.path.join(root, file)
        
        # Gọi hàm solve_edge_feature() trên tệp ảnh
        solve_corner_featur(image_path)