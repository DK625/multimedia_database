import os
from data_process_preparing.same_statio import output_folder

path = os.path.join(output_folder, 'raw_data')

# Lấy danh sách các file trong thư mục và lọc ra các file ảnh (giả sử định dạng ảnh là .jpg)
# files = [f for f in os.listdir(path) if f.endswith('.jpg')]
files = [f for f in os.listdir(path)]

# Sắp xếp các file theo thứ tự tên
files.sort()

# Đổi tên các file ảnh theo dạng 1.jpg, 2.jpg, ...
for index, filename in enumerate(files):
    # Tách phần đuôi file
    file_extension = os.path.splitext(filename)[1]
    # Tạo tên mới cho file
    new_name = f"{index}{file_extension}"
    src = os.path.join(path, filename)
    dst = os.path.join(path, new_name)
    os.rename(src, dst)


print("Đổi tên thành công!")
