#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np
#import tensorflow as tf
import cv2
from tqdm import tqdm
import csv
import numpy as np

path_dataset= r"D:\workspace\multimedia_database\data_process_preparing\delete_background_images"
# data clean là thư mục , trong ddoscos nhiều thư mục con
dir = path_dataset
datadir = path_dataset

a=[len(os.listdir(os.path.join(dir,os.listdir(dir)[x]))) for x in range(0,3)]
# print(a)
sum(a)

Categories = []
# Duyệt qua danh sách các thư mục (lớp) trong datadir (đường dẫn đến dữ liệu hình ảnh), và thêm tên của mỗi thư mục vào danh sách Categories.
for cat in os.listdir(datadir):
    Categories.append(cat)
Categories

dict_cat={}
count=0
for cat in Categories:
  dict_cat[str(count)]=str(cat)
  count+=1

flower_data = []
# img_size = (496, 496)
count=0
for cat in Categories:
    path = os.path.join(datadir, cat)
    class_num = Categories.index(cat)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
        # img_array = cv2.resize(img_array, img_size)
        flower_data.append([ img,img_array])
        print(count, end=" ")
        count=count+1


def rgb_to_hsv(pixel):
    r , g, b = pixel 
    r , g ,b = b / 255.0, g / 255.0, r / 255.0
    
    v = max(r,g,b)
    delta = v - min(r,g,b)
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / v
        if r == v:
            h = (g - b) / delta
        elif g == v:
            h = 2 + (b - r) / delta
        else:
            h = 4 + (r - g) / delta
        h = (h / 6) % 1.0
        
    return [int(h*180), int(s*255), int(v*255)]

def covert_image_rgb_to_hsv(img):
  hsv_image=[]
  for i in img:
    hsv_image2=[]
    for j in i:
      new_color=rgb_to_hsv(j)
      hsv_image2.append((new_color))
    hsv_image.append(hsv_image2)
  hsv_image=np.array(hsv_image)
  return hsv_image

def my_calcHist(image, channels, histSize, ranges):
    # Khởi tạo histogram với tất cả giá trị bằng 0
    hist = np.zeros(histSize, dtype=np.int64)
    # Lặp qua tất cả các pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Lấy giá trị của kênh màu được chỉ định
            bin_vals = [image[i, j, c] for c in channels]
            # Tính chỉ số của bin
            bin_idxs = [(bin_vals[c] - ranges[c][0]) * histSize[c] // (ranges[c][1] - ranges[c][0]) for c in range(len(channels))]
            # Tăng giá trị của bin tương ứng lên 1
            hist[tuple(bin_idxs)] += 1
    return hist

# print(Categories)

data_HSV=[]
for i in range(len(flower_data)) :
  # Đọc ảnh và chuyển đổi sang không gian màu HSV
  img = flower_data[i][1]
  bins = [8,12,3]
  ranges = [[0, 180], [0, 256], [0, 256]]
  img_hsv=covert_image_rgb_to_hsv(img)
  hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
  # print(hist_my.shape)
  embedding = hist_my.flatten()
  embedding[0]=0
  data_HSV.append([flower_data[i][0],embedding])
#   print(i,end=' ')

np.save("hsv1.npy",data_HSV)

# # Tên file CSV
# csv_file = 'hsvcolor_features.csv'

# # Ghi các thuộc tính vào file CSV
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Histogram'])
#     writer.writerow([data_HSV])

# print(f"Dữ liệu đã được ghi vào file: {csv_file}")