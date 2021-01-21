from helpers import *
from feature_extraction import *
from preprocessing import *
warnings.filterwarnings("ignore")

directory='./TestSet'
f_time = open("time.txt", "w")
f_results = open("results.txt", "w")

y_trues=read_Y_TRUE(directory)
counterTest=0
# knn_true=0
svm_true=0

for folder in os.listdir(directory):
    path = directory + '/' + folder+'/'
    if not os.path.isdir(path):
        continue
    # print('##################### - Test '+folder+' - #####################')
    y_true = [y_trues[counterTest]]
    counterTest+=1
    no_of_classes = 3
    
    images, y_trainer = training_data(no_of_classes, path)
    images = np.array(images)

    c = 0
    handwritten_images_binary = []
    scanned_images_binary = []
    handwritten_images_grayscale = []
    scanned_images_grayscale = []
    x_train = np.zeros((len(images), 256))
    
    start = time.time()

    for i in range(len(images)):
        scanned_binary = images[i] < threshold_otsu(images[i])
        scanned_binary, handwritten_binary, scanned_grayscale, handwritten_grayscale = segmentImages(images[i],scanned_binary)
       
        c += 1

        components_handrwitten, lines_handwritten_images, boxes_handrwitten = linesComponents(handwritten_binary, images[i].shape[1])
        if components_handrwitten is None:
            continue

        x_train[i, :] = extract_features(boxes_handrwitten, lines_handwritten_images, len(boxes_handrwitten),handwritten_grayscale)
        # features.append(hist)
        # print('Image Num='+str(i))

###################################################### Training

    # SVM PREDICTION ON TRAINING
    y_train = np.array(y_trainer)
    y_true = np.array(y_true)
    clf_svm= svm.SVC(kernel="linear", C=2, max_iter=1000000)
    clf_svm.fit(x_train, y_train)
    
    # KNN PREDICTION ON TRAINING
    # clf_knn = KNeighborsClassifier(n_neighbors=3)
    # clf_knn.fit(x_train, y_train)
    
###################################################### Load Test Image
    test_image = cv2.imread(path + 'test.png', 0)
    test_images = [test_image]

    scanned_binary = test_image < threshold_otsu(test_image)
    scanned_binary, handwritten_binary, scanned_grayscale, handwritten_grayscale = segmentImages(test_image,scanned_binary)

    c = 0
    x_test = np.zeros((1, 256))
    components_handrwitten, lines_handwritten_images, boxes_handrwitten = linesComponents(
        handwritten_binary, test_image.shape[1])
    if components_handrwitten is None:
        print("No components found in handwritten")

    x_test[0, :] = extract_features(boxes_handrwitten, lines_handwritten_images, len(boxes_handrwitten), handwritten_grayscale)
    
####################################################### Testing
    y_pred_SVM = clf_svm.predict(x_test)
    acc_SVM = metrics.accuracy_score(y_true, y_pred_SVM)
    
    # y_pred_knn = clf_knn.predict(x_test)
    # acc_knn = metrics.accuracy_score(y_true, y_pred_knn)
    
    end = time.time()
    
    # print("Y_true:", y_true)
    # print("Y_pred_SVM:", y_pred_SVM)
    # print("Y_pred_KNN:", y_pred_knn)
    
    svm_true+=acc_SVM
    # knn_true+=acc_knn
    
    f_time.write(str(round(end - start, 2))+'\n')
    f_results.write(str(y_pred_SVM[0])+'\n')
    # print("Time (training+test) taken:", end - start, " seconds")

print("SVM Total Accuracy:", svm_true/counterTest)
f_time.close()
f_results.close()
# print("KNN Total Accuracy:", knn_true/counterTest)
