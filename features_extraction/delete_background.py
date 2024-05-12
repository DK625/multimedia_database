from rembg import remove
from PIL import Image
import os

# inp  = "D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images\\desert_mountains\\free-photo-of-sa-m-c-nui-c-n-c-i-danh-lam-th-ng-c-nh.jpg"

# outp = "D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images\\desert_mountains\\44.jpg"

# input = Image.open(inp)

# output = remove(input)

# # Convert the output image to RGB mode
# output = output.convert("RGB")

# output.save(outp)

input_image_folder = 'D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images\\waterfall_mountains'
output_image_folder = 'D:\\workspace\\multimedia_database\\data_process_preparing\\delete_background_images\\waterfall_mountains'

# Duyệt qua tất cả các thư mục con trong thư mục process_image
for root, dirs, files in os.walk(input_image_folder):
    # Duyệt qua tất cả các tệp trong thư mục hiện tại
    for file in files:
         # Xác định đường dẫn tới tệp ảnh
        image_path = os.path.join(root, file)
        output_path = output_image_folder +  "\\" + file

        input = Image.open(image_path)
        output = remove(input)
        output = output.convert("RGB")
        output.save(output_path)
        # Gọi hàm solve_edge_feature() trên tệp ảnh
        # solve_edge_feature(image_path)