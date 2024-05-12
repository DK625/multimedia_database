# importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

import matplotlib.pyplot as plt

#reading the image
img = imread('D:\\workspace\\multimedia_database\\data_process_preparing\\preprocessed_images\\desert_mountains\\dunes-in-the-sahara-desert-merzouga-morocco-royalty-free-image-1695321159.jpg')
plt.axis("off")
plt.imshow(img)
print(img.shape)

#resizing image
resized_img = resize(img, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img)
plt.show()
print(resized_img.shape)

#creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(fd.shape)
print(hog_image.shape)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()

# save the images
plt.imsave("resized_img.jpg", resized_img)
plt.imsave("hog_image.jpg", hog_image, cmap="gray")