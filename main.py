import cv2 as cv
import numpy as np
import sys
from scipy.ndimage import interpolation as transform
from score import config, filters, io


def import_image(path, size):
    if (len(sys.argv) != 2):
        print("Usage: python3 main.py <filename>")
        print("where <filename> must be in {} directory".format(path))
        exit(0)
    return cv.resize(cv.imread(path + sys.argv[1]), size)


def straighten(img):
    # `img` must be binarized in order for this function to work properly.
    angle = filters.detect_rotation_angle(img)
    img = transform.rotate(img, angle, cval = 255)

    if config.DEBUG:
        print("Detected rotation angle:", angle)
        io.save_and_show("straightened.png", img)

    return img


def detectLines(img):
    img = np.invert(img)

    horizontalSize = int(img.shape[1] / 30)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
    detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations = 1)

    io.save_and_show("detected.jpg", detected_lines)
    return detected_lines


def removeLines(img, lines):
    img = cv.bitwise_or(img, lines)
    io.save_and_show("imgWithoutLines.jpg", img)
    return img


def findBoundingRectangles(img, minArea, maxArea):
    comp = cv.connectedComponentsWithStats(np.invert(img))

    labels = comp[1]
    labelStats = comp[2]
    labelAreas = labelStats[:, 4]

    for compLabel in range(1, comp[0], 1):
        if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
            labels[labels == compLabel] = 0

    labels[labels > 0] = 1

    comp = cv.connectedComponentsWithStats(labels.astype(np.uint8))

    labels = comp[1]
    labelStats = comp[2]

    boxes = []
    newImg = np.ones(img.shape).astype(np.uint8)

    for compLabel in range(1, comp[0], 1):
        x = labelStats[compLabel, 0]
        y = labelStats[compLabel, 1]
        w = labelStats[compLabel, 2]
        h = labelStats[compLabel, 3]
        boxes.append([x, y, w, h])
        score = img[y:y + h, x:x + w]
        newImg[y:y + h, x:x + w] = np.invert(score)

    io.save_and_show("scores.jpg", np.invert(newImg))
    return boxes


def drawBoundingRectangles(img, boxes):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for box in boxes:
        cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv.putText(img, "nuta", (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness = 2)
    io.save_and_show("boxes.jpg", img)


######
img = import_image(path = "img/", size = (800, 500))
gray = filters.desaturate(img)
binary = filters.binarize(gray, block_size = 51, offset = 10)
straight = straighten(binary)
erased = filters.remove_horizontal_lines(straight)
boxes = findBoundingRectangles(erased, minArea = 150, maxArea = 5000)
drawBoundingRectangles(erased, boxes)
######
