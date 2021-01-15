from functions import *

img = cv2.imread('1.png', cv2.IMREAD_COLOR)

img_scanned, img_handwritten = segmentImages(img)
show_images([img_scanned, img_handwritten], ["Scanned", "Hand Written"])
