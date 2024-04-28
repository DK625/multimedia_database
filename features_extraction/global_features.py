import cv2
import pickle
import numpy as np
import os
import pandas as pd

from config import MAIN_ROOT_PATH
from data_process_preparing.data_preparing import preprocessed_images


def segment_image(image):
    # Mô phỏng việc phân đoạn hình ảnh bằng một hàm giả định
    # Trong thực tế, bạn sẽ sử dụng các thuật toán phân đoạn hình ảnh như SegNet, U-Net, Mask R-CNN
    return np.random.rand(image.shape[0], image.shape[1])


def calculate_ratio_and_coordinates(segmented_image):
    # Mô phỏng tính toán tỉ lệ và tọa độ
    aspect_ratio = np.random.rand() * 10  # Tỉ lệ ngẫu nhiên
    coordinates = (np.random.rand() * 100, np.random.rand() * 100)  # Tọa độ ngẫu nhiên
    return aspect_ratio, coordinates


def calculate_geometric_indices(segmented_image):
    # Mô phỏng tính toán các chỉ số hình học
    geometric_indices = {
        'area': np.random.rand() * 1000,  # Diện tích ngẫu nhiên
        'perimeter': np.random.rand() * 100,  # Chu vi ngẫu nhiên
    }
    return geometric_indices


def extract_shape_features(image):
    # Phân đoạn hình ảnh (ví dụ: sử dụng Mask R-CNN)
    segmented_image = segment_image(image)

    # Tính toán tỉ lệ và tọa độ
    aspect_ratio, coordinates = calculate_ratio_and_coordinates(segmented_image)

    # Tính toán chỉ số hình học (ví dụ: diện tích, chu vi)
    geometric_indices = calculate_geometric_indices(segmented_image)

    return aspect_ratio, coordinates, geometric_indices


def calculate_color_histogram(image):
    # Tính toán histogram màu sắc của ảnh
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()


def segment_color_and_calculate_attributes(image):
    # Mô phỏng phân đoạn màu sắc và tính toán thuộc tính
    # Trong thực tế, bạn sẽ sử dụng các kỹ thuật phân đoạn màu sắc và tính toán các thuộc tính cụ thể
    segmented_color_features = {
        'area': np.random.rand() * 1000,  # Diện tích ngẫu nhiên
        'coordinates': (np.random.rand() * 100, np.random.rand() * 100),  # Tọa độ ngẫu nhiên
    }
    return segmented_color_features


def calculate_color_indices(image):
    # Mô phỏng tính toán các chỉ số màu sắc
    color_indices = {
        'mean_color': np.random.rand(3),  # Giá trị màu trung bình ngẫu nhiên
        'color_uniformity': np.random.rand(),  # Độ đồng nhất màu ngẫu nhiên
    }
    return color_indices


def extract_color_features(image):
    # Tính toán histogram màu sắc
    color_histogram = calculate_color_histogram(image)

    # Phân đoạn màu sắc và tính toán các thuộc tính
    segmented_color_features = segment_color_and_calculate_attributes(image)

    # Tính toán các chỉ số màu sắc
    color_indices = calculate_color_indices(image)

    return color_histogram, segmented_color_features, color_indices


def analyze_environment(image):
    # Mô phỏng phân tích đặc trưng môi trường
    # Trong thực tế, bạn sẽ sử dụng các phương pháp xử lý hình ảnh để phân tích sự hiện diện của các đặc trưng môi trường
    environmental_features = {
        'water_presence': np.random.randint(0, 2),  # Sự hiện diện của nước (0 hoặc 1)
        'vegetation_presence': np.random.randint(0, 2),  # Sự hiện diện của cây cỏ (0 hoặc 1)
    }
    return environmental_features


def identify_climate_expression(image):
    # Mô phỏng xác định biểu hiện khí hậu
    # Trong thực tế, bạn sẽ sử dụng các thuật toán nhận dạng mô hình để phân tích thông tin về thời tiết
    climate_expression = 'Sunny'  # Biểu hiện khí hậu (ví dụ: Nắng, Mưa, Mây)
    return climate_expression


def extract_environmental_features(image):
    # Chuyển đổi hình ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tính toán histogram của kênh màu sắc Hue
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])

    # Tính toán số lượng pixel có giá trị lớn hơn ngưỡng cho mỗi kênh màu
    threshold = 150
    green_pixels = np.sum(hsv_image[:, :, 0] > threshold)
    blue_pixels = np.sum(hsv_image[:, :, 1] > threshold)
    red_pixels = np.sum(hsv_image[:, :, 2] > threshold)

    # Tạo một DataFrame để lưu trữ các đặc trưng
    features = pd.DataFrame(columns=['Green Pixels', 'Blue Pixels', 'Red Pixels'])

    # Thêm các giá trị đặc trưng vào DataFrame
    features.loc[0] = [green_pixels, blue_pixels, red_pixels]

    return features


def save_features_to_csv(features, csv_file):
    # Lưu DataFrame vào tệp CSV
    features.to_csv(csv_file, index=False)

# environment_features = extract_environmental_features(image_path)

def extract_features_from_processed_images():
    features = []

    for sub_folder in os.listdir(preprocessed_images):
        sub_folder_path = os.path.join(preprocessed_images, sub_folder)
        if os.path.isdir(sub_folder_path):
            for filename in os.listdir(sub_folder_path):
                image_path = os.path.join(sub_folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Trích xuất đặc trưng từ ảnh
                    shape_features, color_features, environmental_features = extract_features(image)
                    features.append((shape_features, color_features, environmental_features))
                    environment_features = extract_environmental_features(image)
                    save_features_to_csv(environment_features, "environment_features.csv")

    return features


def extract_features(image):
    shape_features = extract_shape_features(image)
    color_features = extract_color_features(image)
    environmental_features = extract_environmental_features(image)
    return shape_features, color_features, environmental_features


# Load extracted features and labels
features = extract_features_from_processed_images()

# storage_path = '/home/doraking/PycharmProjects/multimedia-database/storage'
storage_path = os.path.join(MAIN_ROOT_PATH, 'storage')
# Lưu các đặc trưng vào các tệp tin
with open(os.path.join(storage_path, 'shape_features.pkl'), 'wb') as f:
    pickle.dump([f[0] for f in features], f)

with open(os.path.join(storage_path, 'color_features.pkl'), 'wb') as f:
    pickle.dump([f[1] for f in features], f)

with open(os.path.join(storage_path, 'environmental_features.pkl'), 'wb') as f:
    pickle.dump([f[2] for f in features], f)

print("Number of images processed:", len(features))

# Các hàm extract_shape_features, extract_color_features, và extract_environmental_features được sử dụng để trích xuất các đặc trưng hình dạng,
# màu sắc và môi trường từ ảnh.
# Mỗi hàm có các bước cụ thể để trích xuất các đặc trưng tương ứng, như tính toán tỉ lệ, tọa độ, histogram màu sắc, và phân tích đặc trưng môi trường.
# Các giá trị đặc trưng được in ra để kiểm tra. Trong thực tế, bạn có thể sử dụng các giá trị này để đưa vào mô hình phân loại hoặc nhận diện hình ảnh.
