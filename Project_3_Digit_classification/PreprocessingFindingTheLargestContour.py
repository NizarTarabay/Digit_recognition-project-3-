import os
from PIL import Image
import pandas as pd
import numpy as np
from largest_contour import largest_cnt

def preprocess_training_data (input_path):
    # Prepare the training data
    k = 0
    sos = 40000
    training_set = []
    os.chdir(input_path)
    images = pd.read_pickle(input_path + '/train_images.pkl')
    images = images.reshape(40000, 64, 64)
# --------------------------------------- Pre-processing loop -----------------------------------------#
    for image in images[0:sos]:
        img = Image.fromarray(image)
    # --------------------------- Detecting contours and pre-processing images ------------------------- #
        cropped_img = largest_cnt(img, 230)
    # --------------------------- End Detecting contours and pre-processing images ------------------------- #
        cropped_img_flt = cropped_img.flatten()
        training_set.append(cropped_img_flt)
        k = k+1
    training_set = np.asarray(training_set)
    training_set = training_set.reshape(sos, 64, 64)
    return training_set

def preprocess_test_data (input_path):
    # Prepare the training data
    k = 0
    sos = 10000
    test_set = []
    images = pd.read_pickle(input_path + '/test_images.pkl')
    images = images.reshape(10000, 64, 64)
# --------------------------------------- Pre-processing loop ----------------------------------------- #
    for image in images[0:sos]:
        img = Image.fromarray(image)
    # --------------------------- Detecting contours and pre-processing images ------------------------- #
        cropped_img = largest_cnt(img, 230)
    # --------------------------- End Detecting contours and pre-processing images ------------------------- #
        cropped_img_flt = cropped_img.flatten()
        test_set.append(cropped_img_flt)
        k = k+1
    test_set = np.asarray(test_set)
    test_set = test_set.reshape(sos, 64, 64)
    return test_set
