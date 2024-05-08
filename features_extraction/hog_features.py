import os
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
import cv2
import matplotlib.pyplot as plt
import csv
from skimage import exposure
from skimage import feature

image_folder = 'D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images'

import cv2
def convert_image_rgb_to_gray(img_rgb, resize="no"):
  h, w, _ = img_rgb.shape
  # Create a new grayscale image with the same height and width as the RGB image
  img_gray = np.zeros((h, w), dtype=np.uint32)

  # Convert each pixel from RGB to grayscale using the formula Y = 0.299R + 0.587G + 0.114B
  for i in range(h):
      for j in range(w):
          r, g, b = img_rgb[i, j]
          gray_value = int(0.299*r + 0.587*g + 0.114*b)
          img_gray[i, j] = gray_value
  # print(gray_image.shape())
  if resize!="no":
     img_gray = cv2.resize(src=img_gray, dsize=(496, 496))
  return np.array(img_gray)

def hog_feature(gray_img):# default gray_image
  # 1. Khai báo các tham số
  (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8 , 8),
    cells_per_block=(2,2), transform_sqrt=True, block_norm="L2",
    visualize=True)
  return hog_feats

data_hog=[]

flower_data = []

# Duyệt qua tất cả các thư mục con trong thư mục process_image
for root, dirs, files in os.walk(image_folder):
    # Duyệt qua tất cả các tệp trong thư mục hiện tại
    for file in files:
        # Xác định đường dẫn tới tệp ảnh
        image_path = os.path.join(root, file)
        
        # Đọc ảnh bằng cv2.imread()
        img_array = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        flower_data.append(img_array)

for i in range(len(flower_data)) :
  img_gray=convert_image_rgb_to_gray(flower_data[i])
  embedding=hog_feature(img_gray)
  embedding = embedding.flatten()
  data_hog.append([flower_data[i],embedding])
  print(i,end=' ')

print(len(data_hog))


np.save("hog.npy",data_hog)