import pickle
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import os

from config import MAIN_ROOT_PATH
from data_process_preparing.data_preparing import preprocessed_images


def object_detection_and_feature_extraction(image):
    # Load pre-trained YOLO model
    yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

    # Perform object detection
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)
    outs = yolo.forward(output_layers)

    # Extract features from regions containing detected objects
    features = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get coordinates of the detected object
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])

                # Extract region of interest (ROI)
                roi = image[center_y - h // 2:center_y + h // 2, center_x - w // 2:center_x + w // 2]

                # Resize ROI to fit VGG16 input size
                roi_resized = cv2.resize(roi, (224, 224))

                # Preprocess input for VGG16 model
                roi_preprocessed = preprocess_input(np.expand_dims(roi_resized, axis=0))

                # Extract features using VGG16 model
                vgg16 = VGG16(weights='imagenet', include_top=False)
                model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block5_pool').output)
                features.append(model.predict(roi_preprocessed))

    return features


# Đường dẫn đến thư mục lưu trữ
storage_path = os.path.join(MAIN_ROOT_PATH, 'storage')


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
                    object_features = object_detection_and_feature_extraction(image)
                    features.append((object_features))

    return features


# Load preprocessed image
# preprocessed_image = cv2.imread('preprocessed_image.jpg')

# Perform object detection and feature extraction
object_features = extract_features_from_processed_images()

# Lưu các đặc trưng vào các tệp tin trong thư mục storage
# with open(os.path.join(storage_path, 'object_features.pkl'), 'wb') as f:
#     pickle.dump(object_features, f)

with open(os.path.join(storage_path, 'object_features.pkl'), 'wb') as f:
    pickle.dump([f[2] for f in object_features], f)

print("Number of images processed:", len(object_features))

print("Object Features:", object_features)
