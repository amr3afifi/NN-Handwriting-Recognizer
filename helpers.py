import cv2
import time
import numpy as np
import os
import skimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import glob
import warnings

from sklearn import svm
from sklearn import metrics
from skimage.filters import threshold_otsu
from sklearn.neighbors import KNeighborsClassifier



def show_images(images, titles=None):
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
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


#
# def segmentImages(grayscale_img, Binary_img):
#     width_img = Binary_img.shape[1]
#     length_img = Binary_img.shape[0]
#
#     labels = skimage.measure.label(Binary_img, connectivity=2, return_num=True)
#
#     # fig, ax = plt.subplots(figsize=(10, 6))
#     # ax.imshow(Binary_img)
#
#     regions = skimage.measure.regionprops(labels[0])
#
#     segments = []
#
# for region in regions: minr, minc, maxr, maxc = region.bbox if maxc - minc >= 0.5 * width_img: segments.append([
# minr, minc, maxr, maxc]) #         rect =  mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False,
# edgecolor='red', linewidth=1) #         ax.add_patch(rect)
#
#     # ax.set_axis_off()
#     # plt.tight_layout()
#     # plt.show()
#
#     segmentedImages = []
#
#     for s in segments:
#         minc = s[1]
#         maxc = s[3]
#         minr = s[0]
#         maxr = s[2]
#         seg = [Binary_img[minr:maxr, minc:maxc], minc, maxc, minr, maxr]  # minc,maxc,minr,maxr
#         segmentedImages.append(seg)
#
#     # RE-ORDER IMAGES:
#
#     segmentedImages.sort(key=lambda x: x[3])
#     #     for x in segmentedImages:
#     #         show_images([x[0]])
#     #         print("seg --> " ,x[1:])
#     segmentedImages = np.array(segmentedImages)
#     scanned = Binary_img[segmentedImages[0][4]:segmentedImages[1][4], 20:]
#     handwritten = Binary_img[segmentedImages[1][4]:segmentedImages[2][4], 20:]
#     grayscale_img_scanned = grayscale_img[segmentedImages[0][4]:segmentedImages[1][4], 20:]
#     grayscale_img_handwritten = grayscale_img[segmentedImages[1][4]:segmentedImages[2][4], 20:]
#     return scanned, handwritten, grayscale_img_scanned, grayscale_img_handwritten

#
# # TODO CHECK RETURNS NULL
# def linesComponents(binary_image, originalImageWidth):
#     dilated = binary_dilation(binary_image, np.ones((1, originalImageWidth // 10)))
#     dilated = binary_erosion(dilated, np.ones((5, 1)))
#
#     lines_components, lines_sorted_images, lines_boxes, lines_areas_over_bbox = CCA(dilated, True)
#     if (lines_components is None):
#         return None, None, None
#
#     arrayOfLines, new_boxes = segmentBoxesInImage(lines_boxes, binary_image, True)
#     # displayComponents(binary_image, new_boxes)
#     return lines_components, arrayOfLines, new_boxes
#
#
# def CCA(binary, rowcol):
#     labeled_image = skimage.measure.label(binary, connectivity=2, return_num=True, background=0)
#     try:
#         components = skimage.measure.regionprops(labeled_image[0])
#     except ValueError:  # raised if `y` is empty.
#         return None, None, None, None
#
#     thisdict = {}
#     sorted_segmented_images = []
#     index = 0
#     keys = []
#     boxes = []
#     areas_over_bbox = []
#     # for component in components:
#     #     minR, minC, maxR, maxC = component.bbox
#     #     if rowcol:
#     #         if horizontalBox(component.bbox):
#     #             thisdict[minR] = []
#     #             thisdict[minR].append(binary[minR:maxR + 2, minC:maxC + 2])
#     #             thisdict[minR].append(component.bbox)
#     #             thisdict[minR].append(component.area / component.bbox_area)
#     #             keys.append(str(index))
#     #             index += 1
#     #     else:
#     #         thisdict[minC] = []
#     #         thisdict[minC].append(binary[minR:maxR + 2, minC:maxC + 2])
#     #         thisdict[minC].append(component.bbox)
#     #         thisdict[minC].append(component.area / component.bbox_area)
#     #         keys.append(str(index))
#     #         index += 1
#     if components is not None:
#         thisdict = {component.bbox[0] if rowcol else component.bbox[1]: [
#             binary[component.bbox[0]:component.bbox[2] + 2, component.bbox[1]:component.bbox[3] + 2], component.bbox,
#             component.area / component.bbox_area] for component in components if
#             horizontalBox(component.bbox) or not rowcol}
#         for key in sorted(thisdict.keys()):
#             sorted_segmented_images.append(thisdict[key][0])
#             boxes.append(thisdict[key][1])
#             areas_over_bbox.append(thisdict[key][2])
#     else:
#         # Mafiish components
#         raise ValueError
#     return components, sorted_segmented_images, boxes, areas_over_bbox
#
#
# def segmentBoxesInImage(boxes, image_to_segment, lineword):
#     lines = []
#     new_boxes = []
#     boxat = np.array(boxes)
#     average_area = np.average((boxat[:, 2] - boxat[:, 0]) * (boxat[:, 3] - boxat[:, 1]))
#     if lineword:
#         for box in boxes:
#             [Ymin, Xmin, Ymax, Xmax] = box
#             # For small segmentation errors
#             if (Ymax - Ymin) * (Xmax - Xmin) >= average_area * 0.4:
#                 # rr, cc = rectangle_perimeter(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=image_to_segment.shape)
#                 # notes_with_lines.append(image_to_segment[np.min(rr):np.max(rr), np.min(cc):np.max(cc)])
#                 lines.append(image_to_segment[Ymin:Ymax, Xmin:Xmax])
#                 new_boxes.append(box)
#     else:
#         for box in boxes:
#             [Ymin, Xmin, Ymax, Xmax] = box
#             # For comma and dots
#             if (Ymax - Ymin) * (Xmax - Xmin) >= 350:
#                 rr, cc = rectangle_perimeter(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=image_to_segment.shape)
#                 # TODO reflect the change bta3 el lines hena
#                 # temp_word = image_to_segment[np.min(rr):np.max(rr), np.min(cc):np.max(cc)]
#                 lines.append(image_to_segment[Ymin:Ymax, Xmin:Xmax])
#                 # notes_with_lines.append(temp_word)
#                 new_boxes.append(box)
#
#     # notes_with_lines = theCase(notes_with_lines)
#     lines = np.array(lines)  # , dtype=object)
#     return lines, np.array(new_boxes)

#
# def segmentImages2(rgbimage):
#     # gray = rgb2gray(rgbimage)
#     # show_images([gray])
#
#     # Find the edges in the image using canny detector
#     # edges = canny(gray, 50, 200)
#     # blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(rgbimage, 150, 220)
#     # show_images([edges])
#     # Detect points that form a line
#     # tested_angles = np.array([-np.pi / 2, np.pi / 2])
#     # tested_angles1 = np.linspace(-np.pi / 2, -0.5, 100, endpoint=False)
#     # tested_angles2 = np.linspace(0.5, np.pi / 2, 100, endpoint=False)
#     # tested_angles = np.append(tested_angles1, tested_angles2)
#     tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
#
#     h, theta, d = hough_line(edges, theta=tested_angles)
#     # Generating figure 1
#     # fig, axes = plt.subplots(1, 3, figsize=(15, 6))
#     # ax = axes.ravel()
#     print("el edges ya za3em", edges.shape)
#     # show_images([edges])
#     min_dist = int(0.01 * edges.shape[0]) * 0
#     indices = np.zeros((3, 2))
#     # ax[0].imshow(edges, cmap=cm.gray)
#     # show_images([rgbimage])
#     origin = np.array((0, edges.shape[1]))
#     # print("origin:",origin)
#     i = 0
#     for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_dist, threshold=0.3)):
#         print(angle, dist)
#         if -1 < angle < 1:
#             continue
#         y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#         indices[i][0] = int(y0)
#         indices[i][1] = int(y1)
#         i += 1
#         if i == 3:
#             break
#         # print(y0,y1)
#         # ax[0].plot(origin, (y0, y1), '-r')
#
#     print("Lines indices: ", indices)
#     # ax[0].set_xlim(origin)
#     # ax[0].set_ylim((edges.shape[0], 0))
#     # ax[0].set_axis_off()
#     # ax[0].set_title('Detected lines')
#
#     # plt.tight_layout()
#     # plt.show()
#     indices = np.sort(indices, axis=0)
#     # print(indices)
#     # Show result
#     if indices.any():
#         img_scanned = rgbimage[int(indices[0][0]):int(indices[1][0]), :]
#         img_handwritten = rgbimage[int(indices[1][0]):int(indices[2][0]), :]
#
#         img_scanned = np.negative(img_scanned)
#         img_handwritten = np.negative(img_handwritten)
#     else:
#         img_scanned = None
#         img_handwritten = None
#         print("NONESSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
#     # show_images([img_scanned], ["img scanned"])
#
#     return img_scanned, img_handwritten

#
# def segmentImages3(grayscale_img, Binary_img):
#     width = Binary_img.shape[1]
#
#     labels = skimage.measure.label(Binary_img, connectivity=2, return_num=True)
#
#     components = skimage.measure.regionprops(labels[0])
#
#     segments = [[component.bbox[0], component.bbox[1], component.bbox[2], component.bbox[3]] for component in components
#                 if component.bbox[3] - component.bbox[1] >= 0.5 * width]
#
#     segments.sort(key=lambda x: x[0])
#     scanned = Binary_img[segments[0][2]:segments[1][0], 20:]
#     handwritten = Binary_img[segments[1][2]:segments[2][0], 20:]
#     grayscale_img_scanned = grayscale_img[segments[0][2]:segments[1][0], 20:]
#     grayscale_img_handwritten = grayscale_img[segments[1][2]:segments[2][0], 20:]
#
#     return scanned, handwritten, grayscale_img_scanned, grayscale_img_handwritten
#

# CCA Display Components TESTED
def displayComponents(binary, boxes):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(binary)
    for box in boxes:
        minR, minC, maxR, maxC = box
        rect = mpatches.Rectangle((minC, minR), maxC - minC, maxR - minR, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

#
# # minr, minc, maxr, maxc
# def horizontalBox(box):
#     return ((box[3] - box[1]) / (box[2] - box[0])) > 0.9

#
# # LBP Feature
# def our_lbp(grayscale_img, binary_img):
#     lbp = np.zeros(256)
#     for i in range(1, grayscale_img.shape[0] - 1):
#         for j in range(1, grayscale_img.shape[1] - 1):
#             if (binary_img[i][j] == 0):
#                 continue
#             pixel = grayscale_img[i][j]
#             binary = [int(grayscale_img[i - 1, j - 1] > pixel), int(grayscale_img[i - 1, j] > pixel),
#                       int(grayscale_img[i - 1][j + 1] > pixel), int(grayscale_img[i][j + 1] > pixel),
#                       int(grayscale_img[i + 1][j + 1] > pixel), int(grayscale_img[i + 1][j] > pixel),
#                       int(grayscale_img[i + 1][j - 1] > pixel), int(grayscale_img[i][j - 1] > pixel)]
#             res = int("".join(str(x) for x in binary), 2)
#             lbp[res] += 1
#     return lbp


def read_training_image(path):
    img = cv2.imread(path, 0)
    return img


def training_data(no_of_classes, path):
    x_train = []
    y_train = []
    for i in range(no_of_classes):
        # print("ana hena ya afifif")
        str1 = path
        str2 = i + 1
        str3 = '/*'
        str4 = str1 + str(str2) + str3
        # str4 = str1 + str3
        # print("str4", str4)
        for filename in sorted(glob.glob(str4)):
            # print("filname", filename)
            img = read_training_image(filename)  ## cv2.imread reads images in RGB format
            # show_images([img])
            x_train.append(img)
            y_train.append(i + 1)
    return x_train, y_train

#### wasakha

# def split_dataset():
#     f = open('forms.txt', 'r')
#     words = [word.strip() for word in f.readlines()]
#     # imgs = [cv2.imread(file) for file in glob.glob("JOE/sowary/*.png")]
#     wordSplit = []
#     labels = []
#     print(len(words))
#     for i in range(len(words)):
#         wordSplit.append(words[i].split(" "))
#     for j in range(len(words)):
#         labels.append(wordSplit[j])
#         # for i in range(len(words)):
#         #     print(x[i][0],x[i][1])

#     # x_train = np.asarray(imgs)
#     y_train = np.asarray(labels)
#     # print(y_train)
#     f.close()
#     print(f.closed)
#     arr = []
#     for i in range(len(words)):
#         arr.append(y_train[i][1])
#         print(arr[i])
#         arr[i] = int(arr[i])
#     counts = np.zeros(len(words) )
#     for i in range(len(words)):
#         counts[arr[i]] += 1
#     print("Array of i :", len(arr))
#     print(counts)
#     counter = counts > 1
#     training= []
#     for i in range(len(arr)):
#         if counter[i] == False:
#             training.append(arr[i])
#     print("classes of indices less than 1",arr)
#     indices = [i for i, x in enumerate(counter) if x]
#     print("Len of training:",len(training))

#     # print(indices)
#     x = np.asarray(indices)
#     print(len(x),"ana x")
#     flag = np.zeros(len(x))
#     print(flag)
#     # for j in range(len(words)):
#     #     print(int(y_train[i][1]))
#     for i in range(len(x)):
#         print(x[i])
#     y_true = []
#     y_train_values = []
#     for j in range(y_train.shape[0]):
#         for i in range(len(x)):
#             if j != x[i]:
#                 y_train_values.append(y_train[j, 1])
#     for i in range(len(x)):
#         for j in range(len(words)):
#             if int(y_train[j][1]) == x[i] and flag[i] == 0:
#                 flag[i] = 1
#                 print(y_train[j][0])
#                 y_true.append(int(y_train[j][1]))
#                 print(y_train[j][1])
#                 # Move a file from the directory d1 to d2
#                 shutil.move('training/' + str(y_train[j][0]) + '.png',
#                             'test/' + str(y_train[j][0]) + '.png')
#             # else:# flag[i] !=0:
#             #     y_train_values.append(int(y_train[j][1]))
#             # elif int(y_train[j][1]) != x[i]:
#             #     y_train_values.append(int(y_train[i][1]))
#     # for i in range(len(x)):
#     #     if flag[i] ==0:   
#     #         y_train_values.append(int(y_train[i][1]))
#     # print(len(x))
#     # print(len(y_train_values))
#     # for i in range(len(words)):
#     #     y_train_values.append(int(y_train[i][1]))
#         # print(y_train[i][0], )
#     return y_true, training

# y_true = []
# y_trainer= []
# y_true, y_trainer = split_dataset()

# print(len(y_trainer))

def read_Y_TRUE(directory):
    f = open(directory+'/out.txt', 'r')
    words = [word.strip() for word in f.readlines()]
    words=np.array(words).astype(int)
    return words
    
