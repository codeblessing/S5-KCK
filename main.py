import cv2
import numpy as np
import sys
from scipy.ndimage import interpolation as inter

# ---- Helpers ---- #

def saveAndShow(filename, img, path="out/"):
    cv2.imwrite(path + filename, img)
    cv2.imshow(filename, img)
    cv2.waitKey(0)

def deskew_helper(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return score

# ---- ###### ---- #

def readImageFromTerminal(path, size):
    if(len(sys.argv) != 2):
        print("Usage: python3 main.py <filename>")
        print("where <filename> must be in {} directory".format(path))
        exit(0)
    return cv2.resize(cv2.imread(path + sys.argv[1]), size)


def makeGray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray,None)
    saveAndShow("gray.jpg", gray)
    return gray

def binarize(img, blockSize, offset):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, offset)
    thresh = cv2.bilateralFilter(thresh,9,75,75)
    saveAndShow("thresh.jpg", thresh)
    return thresh

def deskew(bin_img, delta, limit):
    bin_img = np.invert(bin_img)
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        score = deskew_helper(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle:', best_angle)
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = np.invert(np.array(data).astype(np.uint8))
    
    return img


def detectLines(img):
    img = np.invert(img)
    
    horizontalSize = int(img.shape[1] / 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    saveAndShow("detected.jpg", detected_lines)
    return detected_lines

def removeLines(img, lines):
    img = cv2.bitwise_or(img, lines)
    saveAndShow("imgWithoutLines.jpg", img)
    return img

def findBoundingRectangles(img, minArea, maxArea):
    comp = cv2.connectedComponentsWithStats(np.invert(img))

    labels = comp[1]
    labelStats = comp[2]
    labelAreas = labelStats[:,4]

    for compLabel in range(1,comp[0],1):
        if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
            labels[labels==compLabel] = 0

    labels[labels>0] =  1

    comp = cv2.connectedComponentsWithStats(labels.astype(np.uint8))

    labels = comp[1]
    labelStats = comp[2]

    boxes = []
    newImg = np.ones(img.shape).astype(np.uint8)

    for compLabel in range(1,comp[0],1):
        x = labelStats[compLabel,0]
        y = labelStats[compLabel,1]
        w = labelStats[compLabel,2]
        h = labelStats[compLabel,3]
        boxes.append([x,y,w,h])
        score = img[y:y+h, x:x+w]
        newImg[y:y+h, x:x+w] = np.invert(score)
    
    saveAndShow("scores.jpg", np.invert(newImg))
    return boxes


def classify(boxes):
    #TODO
    return []

def drawBoundingRectangles(img, boxes, classes):
    # classes - not used - TODO

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,255,0), 2)
        cv2.putText(img, "nuta", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), thickness=2)
    saveAndShow("boxes.jpg", img)

    
######
img = readImageFromTerminal(path="img/", size=(800, 500))
gray = makeGray(img)
binary = binarize(gray, blockSize=51, offset=10)
deskewed = deskew(binary, delta=1, limit=50)
lines = detectLines(deskewed)
imgWithoutLines = removeLines(deskewed, lines)
boxes = findBoundingRectangles(imgWithoutLines, minArea=150, maxArea=5000)
classes = classify(boxes)
drawBoundingRectangles(deskewed, boxes, classes)
######
