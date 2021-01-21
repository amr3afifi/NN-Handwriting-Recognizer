# LBP Feature

import numpy as np


def our_lbp(grayscale_img, binary_img):
    lbp = np.zeros(256)
    for i in range(1, grayscale_img.shape[0] - 1):
        for j in range(1, grayscale_img.shape[1] - 1):
            if (binary_img[i][j] == 0):
                continue
            pixel = grayscale_img[i][j]
            binary = [int(grayscale_img[i - 1, j - 1] > pixel), int(grayscale_img[i - 1, j] > pixel),
                      int(grayscale_img[i - 1][j + 1] > pixel), int(grayscale_img[i][j + 1] > pixel),
                      int(grayscale_img[i + 1][j + 1] > pixel), int(grayscale_img[i + 1][j] > pixel),
                      int(grayscale_img[i + 1][j - 1] > pixel), int(grayscale_img[i][j - 1] > pixel)]
            res = int("".join(str(x) for x in binary), 2)
            lbp[res] += 1
    return lbp


def extract_features(box, line, length, gray_line):
    hist = np.array([our_lbp(gray_line[box[j][0]:box[j][2], box[j][1]:box[j][3]], line[j])
                     for j in range(length) if box is not None])
    hist = np.average(hist, 0)
    return hist
