from functions import *


def extract_features(image):
    img_scanned, img_handwritten = segmentImages(image)
    if img_scanned is None or img_handwritten is None:
        return None
    show_images([img_scanned, img_handwritten], ["Scanned", "Hand Written"])
    # Binarizing returned images
    scanned_gray = rgb2gray(img_scanned)
    scanned_binary = scanned_gray > threshold_otsu(scanned_gray)

    handwritten_gray = rgb2gray(img_handwritten)
    handwritten_binary = handwritten_gray > threshold_otsu(handwritten_gray)

    # show_images([scanned_binary,handwritten_binary], ["scanned", "handwritten"])

    # Retrieving Lines Images and Components
    scanned_components, scanned_Lines = linesComponents(scanned_binary, image.shape[1])
    handwritten_components, handwritten_Lines = linesComponents(handwritten_binary, image.shape[1])

    print("number of Scanned lines", len(scanned_Lines))

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
    average_word_gaps_line = []
    print("number of handwritten lines", len(handwritten_Lines))
    max_col = 0
    min_row = 0
    line_gaps = []
    for i in range(len(handwritten_Lines)):
        handwritten_words_components, handwritten_arrayOfWords, handwritten_words_boxes = wordsComponents(
            handwritten_Lines[i])
        # f6
        if i != 0:
            line_gaps.append(handwritten_components[i].bbox[2] - min_row)
        min_row = handwritten_components[i].bbox[0]
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
            average_word_gaps_line.append(np.average(words_gaps))
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
    # print("words gaps:", average_word_gaps_line)
    sum_handwritten /= len(handwritten_Lines)
    avg_length_handwritten = np.average(avgs_length_handwritten)
    avg_height_handwritten = np.average(avgs_height_handwritten)
    avgs_blobs_handwritten = np.array(avgs_blobs_handwritten)

    f1 = sum_scanned / sum_handwritten
    f2 = avg_length_scanned / avg_length_handwritten
    f3 = avg_height_scanned / avg_height_handwritten
    f4 = np.average(avgs_blobs_handwritten)
    f5 = np.average(average_word_gaps_line)
    f6 = np.average(line_gaps)

    # Retrieving Words Images and Components from each line
    features = [f1, f2, f3, f4, f5, f6]
    return features


def calculateDistance(x1, x2):
    distance = np.linalg.norm(x1 - x2)
    return distance


def calc_accuracy(knns, true_values, ntest):
    total_predictions = ntest  # np.array(test_images).shape[0]
    correct_knn = 0
    for i in range(len(true_values)):
        if true_values[i] == knns[i]:
            correct_knn += 1
    accuracy_knn = correct_knn / len(true_values)
    print("K-Nearest Neighbour Classifier Accuracy: ", accuracy_knn, "%")
    return accuracy_knn


def KNN(test_point, training_features, y_train, k):
    classification = 0

    minDist = [999999 for i in range(k)]
    minClass = [3 for i in range(k)]

    zero_author = training_features[y_train == 0]
    first_author = training_features[y_train == 1]
    second_author = training_features[y_train == 2]
    third_author = training_features[y_train == 3]
    fourth_author = training_features[y_train == 4]
    fifth_author = training_features[y_train == 5]
    sixth_author = training_features[y_train == 6]
    seventh_author = training_features[y_train == 7]
    eighth_author = training_features[y_train == 8]
    nineth_author = training_features[y_train == 9]
    tenth_author = training_features[y_train == 10]

    for i in zero_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 0
    for i in first_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 1
    for i in second_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 2
    for i in third_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 3
    for i in fourth_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 4
    for i in fifth_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 5
    for i in sixth_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 6
    for i in seventh_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 7
    for i in eighth_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 8
    for i in nineth_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 9
    for i in tenth_author:
        c = calculateDistance(i, test_point)
        if c < max(minDist):
            minDist[minDist.index(max(minDist))] = c
            minClass[minDist.index(max(minDist))] = 10

    # ------------------------------------------------------------------------------------------------------

    zero = minClass.count(0)
    one = minClass.count(1)
    two = minClass.count(2)
    three = minClass.count(3)
    four = minClass.count(4)
    fiveth = minClass.count(5)
    sixth = minClass.count(6)
    seventh = minClass.count(7)
    eighth = minClass.count(8)
    nineth = minClass.count(9)
    tenth = minClass.count(10)

    temp = [zero, one, two, three, four, fiveth, sixth, seventh, eighth, nineth, tenth]
    classification = temp.index(max(temp))
    return classification


def readPipeline(directory):
    images = []
    imageNames = []
    imageClass = []

    for folder in os.listdir(directory):
        insideDir = directory + '/' + folder

        for classNum in os.listdir(insideDir):
            insideDir2 = insideDir + '/' + classNum

            if not os.path.isfile(insideDir2):
                for image in os.listdir(insideDir2):
                    imageDir = insideDir2 + '/' + image

                    if os.path.isfile(imageDir):
                        img = cv2.imread(imageDir, 0)
                        images.append(img)
                        imageNames.append(image)
                        imageClass.append(classNum)

    return images, imageNames, imageClass


def read_training_image(path):
    # img = (io.imread(path))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = cv2.imread(path) # , cv2.IMREAD_COLOR)
    # gray = rgb2gray(io.imread(path))
    # thresh = threshold_sauvola(gray, window_size=61)
    # normalize = gray > thresh
    return img


def predict(test_images, shapes, true_values, training_features, y_train):
    knns = []
    for i in range(len(test_images)):
        # Read each image in the test directory, preprocess it and extract its features.
        img_original = read_training_image(test_images[i])
        show_images([img_original], ["Test image"])
        test_point = extract_features(img_original)

        # Print the actual class of each test figure.
        print("Actual class :", shapes[true_values[i]])
        print("---------------------------------------")

        k = 3
        knn_prediction = KNN(test_point, training_features, y_train, k)
        knns.append(knn_prediction)

        print("K-Nearest Neighbours Prediction          :", shapes[knn_prediction])
        print("===========================================================================")
        '''
        # Visualize each test figure.
        fig = plt.figure()
        plt.imshow(img_original)
        plt.axis("off")
        plt.show()
        '''

    return knns


def training_data(shapes):
    x_train = []
    y_train = []
    for i in range(len(shapes)):
        # print("ana hena ya afifif")
        str1 = 'larger_test_set/'
        str2 = i
        str3 = '/*'
        str4 = str1 + str(str2) + str3
        print("str4", str4)
        for filename in sorted(glob.glob(str4)):
            print("filname", filename)
            img = read_training_image(filename)  ## cv2.imread reads images in RGB format
            # show_images([img])
            x_train.append(img)
            y_train.append(i)
    return x_train, y_train


shapes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
x_train, y_train = training_data(shapes)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
# show_images(x_train)
print("x_train shape:", x_train.shape)
print("y_train:", y_train)
number_of_features = 6
training_features = np.zeros((x_train.shape[0], number_of_features))

for i in range(training_features.shape[0]):
    show_images([x_train[i]],["inside loop"])
    print(x_train[i].shape)
    features = extract_features(x_train[i])
    if features is not None:
        print(features)
        training_features[i, :] = features
    else:
        break
# D:\College\Semester 9\Pattern Recognition and Neural Networks\Project\NN-Handwriting-Recognizer\testdataset\data\01
test_images = sorted(glob.glob('larger_test_set/test/*'))
ntest = len(test_images)
#
true_values = [8, 5, 10, 7, 1]
#
if training_features.any():
    knns = predict(test_images, shapes, true_values, training_features, y_train)
    accuracy_knn = calc_accuracy(knns, true_values, ntest)
else:
    print("akhhhhh")
#

