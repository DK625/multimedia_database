import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import feature
from data_process_preparing.same_statio import output_folder
from features_extraction.features_extract import extract_features
from features_extraction.hsv import covert_image_rgb_to_hsv, my_calcHist
from utils.utils import extract_number

# Đường dẫn tới thư mục chứa ảnh
path = os.path.join(output_folder, 'rock_mountains')


def convert_image_rgb_to_gray(img_rgb):
    h, w, _ = img_rgb.shape
    img_gray = np.zeros((h, w), dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            r, g, b = img_rgb[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            img_gray[i, j] = gray_value
    return img_gray


def hog_feature(gray_img):
    (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                        visualize=True)
    return hog_feats


# Đọc ảnh đầu tiên từ đường dẫn
img_name = sorted(os.listdir(path), key=extract_number)[0]


# img_path = os.path.join(path, img_name)
# img_path_res = os.path.join(path, img_name)


def find_images(img_path_res):
    img = cv2.imread(img_path_res, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (496, 496))

    # plt.imshow(img)
    # plt.title(f"Image: {img_name}")
    # plt.axis('off')  # Ẩn trục tọa độ
    # plt.show()

    # extract_features()

    # Trích xuất đặc trưng HSV
    img_hsv = covert_image_rgb_to_hsv(img)
    bins = [8, 12, 3]
    ranges = [[0, 180], [0, 256], [0, 256]]
    hist_hsv = my_calcHist(img_hsv, [0, 1, 2], bins, ranges).flatten()

    # Trích xuất đặc trưng HOG
    img_gray = convert_image_rgb_to_gray(img)
    hist_hog = hog_feature(img_gray).flatten()

    # Kết hợp đặc trưng HSV và HOG
    input_features = np.concatenate((hist_hsv, hist_hog))

    # Đọc các đặc trưng đã lưu
    data_file = np.load("concat_hog2_hsv2.npy.npy", allow_pickle=True)

    # Tính khoảng cách và tìm 3 ảnh gần nhất
    distances = [np.linalg.norm(input_features - features) for features in data_file]
    nearest_indices = np.argsort(distances)[:4]
    # nearest_indices = np.argsort(distances)[1:4]  # Bỏ qua chỉ số 0

    # Hiển thị kết quả
    print(f"Ba ảnh giống nhất với ảnh đầu vào là: {nearest_indices}")
    return nearest_indices
