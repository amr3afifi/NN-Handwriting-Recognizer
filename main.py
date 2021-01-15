from functions import *

# Read
img = cv2.imread('1.png', cv2.IMREAD_COLOR)

rgbimage = img.copy()
img_scanned, img_handwritten = segmentImages(rgbimage)
show_images([img_scanned, img_handwritten], ["Scanned", "Hand Written"])
