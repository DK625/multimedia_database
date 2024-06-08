import numpy as np

from features_extraction.hog import generate_hog_file
from features_extraction.hsv import generate_hsv_file


def extract_features():
    generate_hsv_file()
    generate_hog_file()
    # len(data_hog)
    data_file_hog = np.load("hog2.npy", allow_pickle=True)
    # len(data_file_hog)

    data_file_hsv = np.load("HSV.npy", allow_pickle=True)

    # trich xuat HSV+HOG

    array_concat_hog_hsv = []
    for i in range(len(data_file_hog)):
        concat_in_value = np.concatenate((data_file_hsv[i], data_file_hog[i]))
        array_concat_hog_hsv.append(concat_in_value)
    np.save("concat_hog2_hsv2.npy", array_concat_hog_hsv)
    print('created concat_hog2_hsv2.npy')

# extract_features()