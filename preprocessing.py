import numpy as np
import skimage
from skimage.morphology import binary_erosion, binary_dilation
from skimage.draw import rectangle_perimeter


def segmentImages(grayscale_img, Binary_img):
    width = Binary_img.shape[1]
    labels = skimage.measure.label(Binary_img, connectivity=2, return_num=True)
    components = skimage.measure.regionprops(labels[0])
    segments = [[component.bbox[0], component.bbox[1], component.bbox[2], component.bbox[3]] for component in components
                if component.bbox[3] - component.bbox[1] >= 0.5 * width]
    segments.sort(key=lambda x: x[0])
    scanned = Binary_img[segments[0][2]:segments[1][0], 20:]
    handwritten = Binary_img[segments[1][2]:segments[2][0], 20:]
    grayscale_img_scanned = grayscale_img[segments[0][2]:segments[1][0], 20:]
    grayscale_img_handwritten = grayscale_img[segments[1][2]:segments[2][0], 20:]

    return scanned, handwritten, grayscale_img_scanned, grayscale_img_handwritten


def linesComponents(binary_image, originalImageWidth):
    dilated = binary_dilation(binary_image, np.ones((1, originalImageWidth // 10)))
    dilated = binary_erosion(dilated, np.ones((5, 1)))
    lines_components, lines_sorted_images, lines_boxes, lines_areas_over_bbox = CCA(dilated, True)
    if lines_components is None:
        return None, None, None
    arrayOfLines, new_boxes = segmentBoxesInImage(lines_boxes, binary_image, True)
    return lines_components, arrayOfLines, new_boxes


# minr, minc, maxr, maxc
def horizontalBox(box):
    return ((box[3] - box[1]) / (box[2] - box[0])) > 0.9


def CCA(binary, rowcol):
    labeled_image = skimage.measure.label(binary, connectivity=2, return_num=True, background=0)
    try:
        components = skimage.measure.regionprops(labeled_image[0])
    except ValueError:  # raised if `y` is empty.
        return None, None, None, None
    thisdict = {}
    sorted_segmented_images = []
    index = 0
    keys = []
    boxes = []
    areas_over_bbox = []
    if components is not None:
        thisdict = {component.bbox[0] if rowcol else component.bbox[1]: [
            binary[component.bbox[0]:component.bbox[2] + 2, component.bbox[1]:component.bbox[3] + 2], component.bbox,
            component.area / component.bbox_area] for component in components if
            horizontalBox(component.bbox) or not rowcol}
        for key in sorted(thisdict.keys()):
            sorted_segmented_images.append(thisdict[key][0])
            boxes.append(thisdict[key][1])
            areas_over_bbox.append(thisdict[key][2])
    else:
        raise ValueError
    return components, sorted_segmented_images, boxes, areas_over_bbox


def segmentBoxesInImage(boxes, image_to_segment, lineword):
    lines = []
    new_boxes = []
    boxat = np.array(boxes)
    average_area = np.average((boxat[:, 2] - boxat[:, 0]) * (boxat[:, 3] - boxat[:, 1]))
    if lineword:
        for box in boxes:
            [Ymin, Xmin, Ymax, Xmax] = box
            if (Ymax - Ymin) * (Xmax - Xmin) >= average_area * 0.4:
                lines.append(image_to_segment[Ymin:Ymax, Xmin:Xmax])
                new_boxes.append(box)
    else:
        for box in boxes:
            [Ymin, Xmin, Ymax, Xmax] = box
            # For comma and dots
            if (Ymax - Ymin) * (Xmax - Xmin) >= 350:
                rr, cc = rectangle_perimeter(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=image_to_segment.shape)
                lines.append(image_to_segment[Ymin:Ymax, Xmin:Xmax])

                new_boxes.append(box)
    lines = np.array(lines)
    return lines, np.array(new_boxes)

