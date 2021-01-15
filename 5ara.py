import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from skimage.filters import median
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, thin
from skimage.exposure import histogram
import skimage
from skimage.draw import rectangle, rectangle_perimeter
import math
import cv2
import numpy as np
from skimage.measure import find_contours
from scipy.signal import find_peaks
from skimage.filters import median, gaussian, threshold_otsu

from skimage.color import rgb2gray, rgb2hsv, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny, match_template
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_local, threshold_niblack
from skimage.transform import hough_line, hough_line_peaks
from skimage.util import random_noise
from skimage.measure import find_contours
from skimage.draw import rectangle, rectangle_perimeter
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.pyplot import bar
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math as m
import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from scipy import fftpack
from scipy.signal import convolve2d
from skimage.util import random_noise
from skimage.filters import median, gaussian, threshold_otsu
from skimage.filters import roberts, sobel, sobel_h, scharr
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, skeletonize, thin
from skimage import img_as_ubyte
from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import skimage.io as io

import matplotlib.patches as mpatch


def deskew(binary):
    blur = gaussian(binary, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 9999 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_angle = histo[1][np.argmax(histo[0])]
    rotation_angle = int(rotation_angle)
    print(rotation_angle)
    '''
    if rotation_angle > 45:
        rotation_number = -(90 - rotation_angle)
    elif rotation_angle < -45:
        rotation_number = 90 - abs(rotation_angle)
    '''
    # show_images([gray,normalize,edges], ["gray", "bin", "edges"])
    rotated = rotate((np.logical_not(normalize)), rotation_angle, resize=True,
                     mode='constant', cval=0).astype(np.uint8)
    # show_images([gray])
    gray = rotate(gray, rotation_angle, resize=True, mode='constant', cval=255)
    return rotated, gray



def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


image = io.imread('1.png')

thresh = threshold_otsu(rgb2gray(image))

normalize = image > thresh
#normalize = np.where(image > thresh, 255.0, 0.0)
print(normalize)
show_images([normalize])
