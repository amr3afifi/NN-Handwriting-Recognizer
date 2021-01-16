from functions import *

# Read
img = cv2.imread('a01-003.png', cv2.IMREAD_COLOR)

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
show_images(scanned_Lines)

sum_scanned = 0
sum_handwritten = 0
avgs_length_scanned = []
avgs_height_scanned = []
avgs_length_handwritten = []
avgs_height_handwritten = []
avgs_blobs_handwritten = []

# Retrieving Words Images and Components from each line

print("number of Scanned lines", len(scanned_Lines))
for i in range(len(scanned_Lines)):
    scanned_words_components, scanned_arrayOfWords, scanned_words_boxes = wordsComponents(scanned_Lines[i])
    # words_images = segmentBoxesInImage(scanned_words_boxes, scanned_Lines[i], False)

    # words_blobs_areas = [blobArea(words_images[j] ^ segmentation.flood_fill(words_images[j], (0, 0), 255)) for j in \
    #                     range(words_images.shape[0])]
    npsum = np.sum(scanned_words_boxes, axis=0)
    length_scanned_words = len(scanned_arrayOfWords)
    avgs_length_scanned.append((npsum[3] - npsum[1]) / length_scanned_words)
    avgs_height_scanned.append((npsum[2] - npsum[0]) / length_scanned_words)
    # avgs_blobs_scanned.append(np.average(words_blobs_areas))
    sum_scanned += length_scanned_words

sum_scanned /= len(scanned_Lines)
avg_length_scanned = np.average(avgs_length_scanned)
avg_height_scanned = np.average(avgs_height_scanned)
average_gaps_line = []
print("number of handwritten lines", len(handwritten_Lines))
max_col = 0
min_col = 0
for i in range(len(handwritten_Lines)):
    handwritten_words_components, handwritten_arrayOfWords, handwritten_words_boxes = wordsComponents(
        handwritten_Lines[i])
    # f5
    words_gaps = []
    words_blobs_areas = []
    if handwritten_arrayOfWords.shape[0] > 1:
        for j in range(handwritten_arrayOfWords.shape[0]):
            # f5
            if j != 0:
                words_gaps.append(handwritten_words_boxes[j][1] - max_col)
            max_col = handwritten_words_boxes[j][3]
            # f4
            words_blobs_areas.append(blobArea(
                handwritten_arrayOfWords[j] ^ segmentation.flood_fill(handwritten_arrayOfWords[j], (0, 0), 255)))
        average_gaps_line.append(np.average(words_gaps))
        avgs_blobs_handwritten.append(np.average(words_blobs_areas))
    # f4 calculations
    # words_blobs_areas = [
    #    blobArea(handwritten_arrayOfWords[j] ^ segmentation.flood_fill(handwritten_arrayOfWords[j], (0, 0), 255)) for j
    #    in \
    #    range(handwritten_arrayOfWords.shape[0])]
    # f1
    length_handwritten_words = len(handwritten_arrayOfWords)
    sum_handwritten += length_handwritten_words
    # f2
    npsum = np.sum(handwritten_words_boxes, axis=0)
    avgs_length_handwritten.append((npsum[3] - npsum[1]) / length_handwritten_words)
    # f3
    avgs_height_handwritten.append((npsum[2] - npsum[0]) / length_handwritten_words)
print("words gaps:", average_gaps_line)
sum_handwritten /= len(handwritten_Lines)
avg_length_handwritten = np.average(avgs_length_handwritten)
avg_height_handwritten = np.average(avgs_height_handwritten)
avgs_blobs_handwritten = np.array(avgs_blobs_handwritten)

f1 = sum_scanned / sum_handwritten
f2 = avg_length_scanned / avg_length_handwritten
f3 = avg_height_scanned / avg_height_handwritten
f4 = np.average(avgs_blobs_handwritten)
f5 = np.average(average_gaps_line)
print("f5", f5)
print("f4", f4)

print(f3)
