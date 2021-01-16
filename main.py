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
show_images(scanned_Lines)

f1=0
sum_scanned=0
sum_handwritten=0
avgs_length_scanned=[]
avgs_height_scanned=[]
avgs_length_handwritten=[]
avgs_height_handwritten=[]
# Retrieving Words Images and Components from each line
for i in range(len(scanned_Lines)):
    scanned_words_components, scanned_arrayOfWords, scanned_words_boxes = wordsComponents(scanned_Lines[i])
    words_images = segmentBoxesInImage(scanned_words_boxes,scanned_Lines[i] , False)
    # TODO CALCULATE THE BLOBS AVERAGE  FOR EVERY IMAGE AND CREATE A FUNCTION  
    example_filled = segmentation.flood_fill(words_images[0],(0,0),255);
    show_images([ example_filled ^ words_images[0],example_filled, words_images[0] ], ["Filled image of word ","Filled image","Original word"])
    npsum=np.sum(scanned_words_boxes,axis=0)
    
    length_scanned_words=len(scanned_arrayOfWords)
    avgs_length_scanned.append((npsum[3]-npsum[1])/length_scanned_words)
    avgs_height_scanned.append((npsum[2] - npsum[0]) / length_scanned_words)
    sum_scanned += length_scanned_words
    

sum_scanned /= len(scanned_Lines)
avg_length_scanned=np.average(avgs_length_scanned)
avg_height_scanned=np.average(avgs_height_scanned)


for line in handwritten_Lines:
    handwritten_words_components, handwritten_arrayOfWords, handwritten_words_boxes = wordsComponents(line)
    npsum = np.sum(handwritten_words_boxes, axis=0)
    length_handwritten_words = len(handwritten_arrayOfWords)
    avgs_length_handwritten.append((npsum[3] - npsum[1]) / length_handwritten_words)
    avgs_height_handwritten.append((npsum[2] - npsum[0]) / length_handwritten_words)
    sum_handwritten += length_handwritten_words

sum_handwritten/=len(handwritten_Lines)
avg_length_handwritten=np.average(avgs_length_scanned)
avg_height_handwritten=np.average(avgs_height_scanned)

f1=sum_scanned/sum_handwritten
f2=avg_length_scanned/avg_length_handwritten
f3=avg_height_scanned/avg_height_handwritten
# f4= 


print(f3)
