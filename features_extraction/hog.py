import os
import numpy as np
from skimage import feature
import cv2

from data_process_preparing.same_statio import output_folder
from utils.utils import extract_number

# path = 'D:\\workspace\\multimedia_database\\data_process_preparing\\delete_background_images\\rock_mountains'
# path = output_folder  # same_statio folder
path = os.path.join(output_folder, 'rock_mountains')


# ham trich xuat dac trung hinh dang HOG


def convert_image_rgb_to_gray(img_rgb, resize="no"):
    h, w, _ = img_rgb.shape
    # Create a new grayscale image with the same height and width as the RGB image
    img_gray = np.zeros((h, w), dtype=np.uint32)

    # Convert each pixel from RGB to grayscale using the formula Y = 0.299R + 0.587G + 0.114B
    for i in range(h):
        for j in range(w):
            r, g, b = img_rgb[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            img_gray[i, j] = gray_value
    # print(gray_image.shape())
    if resize != "no":
        img_gray = cv2.resize(src=img_gray, dsize=(496, 496))
    return np.array(img_gray)


def hog_feature(gray_img):  # default gray_image
    # 1. Khai báo các tham số
    (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                        visualize=True)
    return hog_feats


# trich xuat HOG
def generate_hog_file():
    flower_data = []
    # img_size = (496, 496)
    for img in sorted(os.listdir(path), key=extract_number):
        img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
        # img_array = cv2.resize(img_array, img_size)
        # flower_data.append([ img,img_array])
        flower_data.append(img_array)

    # for i in range(0, 20):
    #     print(flower_data[i][0])

    data_hog = []
    for i in range(len(flower_data)):
        # Đọc ảnh và chuyển đổi sang không gian màu HSV

        # img_hsv=covert_image_rgb_to_hsv(img)
        # hist_my = my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
        # print(hist_my.shape)
        img_gray = convert_image_rgb_to_gray(flower_data[i])
        embedding = hog_feature(img_gray)
        embedding = embedding.flatten()
        # embedding[0]=0
        data_hog.append(embedding)
        print(i, end=' ')

    np.save("hog2.npy", data_hog)
    print('created hog2.npy')


# generate_hog_file()
# len(data_hog)
# data_file_hog = np.load("hog.npy", allow_pickle=True)
# len(data_file_hog)
#
# data_file_hsv = np.load("hsv.npy", allow_pickle=True)
#
# # trich xuat HSV+HOG
#
# array_concat_hog_hsv = []
# for i in range(len(data_file_hog)):
#     concat_in_value = np.concatenate((data_file_hsv[i], data_file_hog[i]))
#     array_concat_hog_hsv.append(concat_in_value)
