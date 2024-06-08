import cv2
import os
from rembg import remove
from PIL import Image


def delete_background(image):
    input_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output_pil = remove(input_pil)
    output = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    return output


def resize_image(image_path, target_size=(500, 500)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không thể đọc file ảnh từ đường dẫn: {image_path}")
    resized_image = cv2.resize(image, target_size)
    return resized_image



def preprocess_image(image_path):
    image_resized = resize_image(image_path)
    image_deleted_background = remove(image_resized)
    return image_deleted_background


def preprocess_images():
    if not os.path.exists(preprocessed_images):
        os.makedirs(preprocessed_images)

    for filename in os.listdir(raw_data_folder):
        image_path = os.path.join(raw_data_folder, filename)
        preprocessed_image = preprocess_image(image_path)
        cv2.imwrite(os.path.join(preprocessed_images, filename), preprocessed_image)


ROOT_PATH = os.path.dirname(__file__)
raw_data_folder = os.path.join(ROOT_PATH, 'raw_data', 'new')
preprocessed_images = os.path.join(ROOT_PATH, 'delete_background_images', 'new')
# preprocess_images()

# python3 data_process_preparing/data_preparing.py
