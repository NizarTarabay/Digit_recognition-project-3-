import os
from PIL import Image
import pandas as pd
import scipy.misc
from largest_contour import largest_cnt
input_path = 'C:/Users/User/Desktop/Project_3_Digit_classification'

# Save training data
k = 0
i = 0
sos = 40000
training_set = []
os.chdir(input_path)
images = pd.read_pickle(input_path + '/train_images.pkl')
images = images.reshape(40000, 64, 64)
train_labels = pd.read_csv(input_path + '/train_labels.csv')
# to save the training set in separate files according to their classes
path_train = input_path+'/Visualize digits/train_64x64('+str(sos)+')/'
if not os.path.exists(path_train):
        os.mkdir(path_train)
# pre-processing loop
for image in images[0:sos]:
    img = Image.fromarray(image)
# ----------------------------------- Detecting contours and pre-processing images ----------------------------------- #
    cropped_img = largest_cnt(img, 230)
# -------------------------------- End Detecting contours and pre-processing images --------------------------------- #
    cropped_img_flt = cropped_img.flatten()
    training_set.append(cropped_img_flt)
    image = cropped_img
    label = str(train_labels._get_values[k, 1])
    sub_path_train = path_train + label

    if not os.path.exists(sub_path_train):
        os.mkdir(sub_path_train)
        print("Directory ",  sub_path_train,  " Created ")
    os.chdir(sub_path_train)
    scipy.misc.imsave(str(k)+'.jpg', image)
    k = k + 1

# Save testing data
k = 0
i = 0
sos = 10000
test_set = []
images = pd.read_pickle(input_path + '/test_images.pkl')
images = images.reshape(10000, 64, 64)
# to save the training set in separate files according to their classes
path_test = input_path+'/Visualize digits/test_64x64('+str(sos)+')/'
if not os.path.exists(path_test):
    os.mkdir(path_test)
    # pre-processing loop
for image in images[0:sos]:
    img = Image.fromarray(image)
# ----------------------------------- Detecting contours and pre-processing images ----------------------------------- #
    cropped_img = largest_cnt(img, 230)
# ------------------------------- End Detecting contours and pre-processing images --------------------------------- #
    cropped_img_flt = cropped_img.flatten()
    test_set.append(cropped_img_flt)
    image = cropped_img
    os.chdir(path_test)
    scipy.misc.imsave(str(k)+'.jpg', image)
    k = k+1
