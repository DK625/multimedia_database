import cv2
import os


def preprocess_image(image_path, target_size=(500, 500)):
    # only resize
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    return resized_image


def preprocess_images():
    if not os.path.exists(preprocessed_images):
        os.makedirs(preprocessed_images)

    for sub_folder in os.listdir(raw_data_folder):
        sub_folder_path = os.path.join(raw_data_folder, sub_folder)
        if not os.path.exists(os.path.join(preprocessed_images, sub_folder)):
            os.makedirs(os.path.join(preprocessed_images, sub_folder))
        if os.path.isdir(sub_folder_path):
            for filename in os.listdir(sub_folder_path):
                image_path = os.path.join(sub_folder_path, filename)
                preprocessed_image = preprocess_image(image_path)
                cv2.imwrite(os.path.join(preprocessed_images, sub_folder, filename), preprocessed_image)


ROOT_PATH = os.path.dirname(__file__)
raw_data_folder = os.path.join(ROOT_PATH, 'raw_data')
preprocessed_images = os.path.join(ROOT_PATH, 'preprocessed_images')
# preprocess_images()
