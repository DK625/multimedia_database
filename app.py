import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from skimage import feature
from matplotlib import pyplot as plt
from data_process_preparing.same_statio import output_folder
from features_extraction.features_extract import extract_features
from features_extraction.hsv import covert_image_rgb_to_hsv, my_calcHist
from find_images import find_images
from utils.utils import extract_number

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['IMAGE_FOLDER'] = 'static/rock_mountains'

# Đường dẫn tới thư mục chứa ảnh
path = os.path.join(output_folder, 'rock_mountains')


def convert_image_rgb_to_gray(img_rgb):
    h, w, _ = img_rgb.shape
    img_gray = np.zeros((h, w), dtype=np.uint8)
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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # img = cv2.imread(file_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (496, 496))
            #
            # # Trích xuất đặc trưng HSV
            # img_hsv = covert_image_rgb_to_hsv(img)
            # bins = [8, 12, 3]
            # ranges = [[0, 180], [0, 256], [0, 256]]
            # hist_hsv = my_calcHist(img_hsv, [0, 1, 2], bins, ranges).flatten()
            #
            # # Trích xuất đặc trưng HOG
            # img_gray = convert_image_rgb_to_gray(img)
            # hist_hog = hog_feature(img_gray).flatten()
            #
            # # Kết hợp đặc trưng HSV và HOG
            # input_features = np.concatenate((hist_hsv, hist_hog))
            #
            # # Đọc các đặc trưng đã lưu
            # data_file = np.load("concat_hog2_hsv2.npy.npy", allow_pickle=True)
            #
            # # Tính khoảng cách và tìm 3 ảnh gần nhất
            # distances = [np.linalg.norm(input_features - features) for features in data_file]
            # nearest_indices = np.argsort(distances)[:3]
            nearest_indices = find_images(file_path)

            # Lấy tên các ảnh gần nhất
            sorted_path = sorted(os.listdir(path), key=extract_number)
            nearest_images = [sorted_path[i] for i in nearest_indices]

            return render_template('result.html', nearest_images=nearest_images, uploaded_image=file.filename)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
