import numpy as np
import cv2

def largest_cnt (img, thr):
    # --------------------------- Detecting contours and pre-processing images ------------------------- #
    ret, thresh = cv2.threshold(np.uint8(img), 230, 255, cv2.THRESH_BINARY)
    # detect the largest contour
    contours, hierarchy = cv2.findContours(np.uint8(255 - thresh), 1, 2)
    largest_area = 0
    i = 0
    area_i = np.array([])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = max(w, h)
        if area > largest_area:
            largest_area = area
            area_i = np.append(area_i, i)
        i = i + 1
    if area_i.size == 1:
        cnt = contours[int(area_i[-1])]
    else:
        cnt = contours[int(area_i[-2])]
    x, y, w, h = cv2.boundingRect(cnt)
    b = np.zeros((64, 64))
    for i in range(0, 64):
        for j in range(0, 64):
            if i > y and i < y + h and j < x + w and j > x:
                b[i, j] = 1
    cropped_img = b * img
    cropped_img = np.uint8(cropped_img)
    ret, cropped_img = cv2.threshold(cropped_img, thr, 255, cv2.THRESH_BINARY)  # change the threshold here
    # --------------------------- End Detecting contours and pre-processing images ------------------------- #
    return cropped_img