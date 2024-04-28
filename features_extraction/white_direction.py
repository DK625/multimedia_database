import numpy as np
import cv2


def compute_gradient(image):
    # Chuyển đổi ảnh sang ảnh độ xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color Image", image)
    cv2.imshow("Gray Image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Tính gradient theo chiều ngang và dọc
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Tính độ lớn của gradient
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Tính hướng của gradient
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    return magnitude, gradient_direction


def find_white_direction(image):
    # Tính toán gradient
    magnitude, gradient_direction = compute_gradient(image)

    # Xác định hướng lớn nhất
    max_direction = np.argmax(magnitude)

    # Lấy hướng gradient tại điểm có độ lớn lớn nhất
    max_gradient_direction = gradient_direction.flatten()[max_direction]

    # Xác định hướng màu trắng
    if np.abs(max_gradient_direction) < np.pi / 4:
        return "Ngang"
    else:
        return "Dọc"


# Đọc ảnh đồi núi
# image_path = "water.jpg"
image_path = "rock.jpg"
image = cv2.imread(image_path)

# Xác định hướng màu trắng trên ảnh
white_direction = find_white_direction(image)
print("Hướng màu trắng trên ảnh là:", white_direction)
