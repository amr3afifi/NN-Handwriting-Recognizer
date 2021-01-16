from functions import *

# Read
img = cv2.imread('1.png', cv2.IMREAD_COLOR)

# Copying the rgb image
rgbimage = img.copy()

# Segmenting image
img_scanned, img_handwritten = segmentImages(rgbimage)
show_images([img_scanned, img_handwritten], ["Scanned", "Hand Written"])

# Binarizing returned images
scanned_gray = rgb2gray(img_scanned)
scanned_binary = scanned_gray > threshold_otsu(scanned_gray)

handwritten_gray = rgb2gray(img_handwritten)
handwritten_binary = handwritten_gray > threshold_otsu(handwritten_gray)

# Retrieving Lines Images and Components
scanned_components, scanned_Lines = linesComponents(scanned_binary, img.shape[1])
handwritten_components, handwritten_Lines = linesComponents(handwritten_binary, img.shape[1])

