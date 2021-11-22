import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def saveAndShow(filename, img, path="out/"):
    cv2.imwrite(path + filename, img)
    cv2.imshow(filename, img)
    cv2.waitKey(0)

def getGrayImage(filename, path="img/", size=(800, 500)):
    img = cv2.resize(cv2.imread(path + filename), size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray,None)
    saveAndShow("gray.jpg", gray)
    return gray

def binarize(img, blockSize, offset):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, offset)
    thresh = cv2.bilateralFilter(thresh,9,75,75)
    saveAndShow("thresh.jpg", thresh)
    return thresh


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return score

def deskew(bin_img):
    bin_img = np.invert(bin_img)
    delta = 1
    limit = 50
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        score = find_score(bin_img, angle)
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

def drawLines(img, line):
    img = cv2.bitwise_or(img, lines)
    saveAndShow("imgLines.jpg", img)
    return img

def splitScores(img):
    maxArea = 5000
    minArea = 100

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
    scores = []
    newImg = np.ones(img.shape).astype(np.uint8)
    strElement = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    for compLabel in range(1,comp[0],1):
        x = labelStats[compLabel,0]
        y = labelStats[compLabel,1]
        w = labelStats[compLabel,2]
        h = labelStats[compLabel,3]
        boxes.append([x,y,w,h])
        score = img[y:y+h, x:x+w]
        score = cv2.morphologyEx(score, cv2.MORPH_OPEN, strElement)
        scores.append(cv2.resize(score, (20,30)))
        newImg[y:y+h, x:x+w] = np.invert(score)
    
    saveAndShow("scores.jpg", np.invert(newImg))
    saveAndShow("flat.jpg", cv2.hconcat(scores))
    return scores, boxes


def classify(scores):
    trainImage = cv2.cvtColor(cv2.imread("templates/flat2.jpg"), cv2.COLOR_BGR2GRAY)
    trainImage = cv2.adaptiveThreshold(trainImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    trainImage = trainImage.reshape(-1, trainImage.shape[0] * trainImage.shape[1]).astype(np.float32)

    train = np.array(np.split(trainImage[0], 32)).astype(np.float32)
    train_labels = [0,0,0,0,0,1,1,1,1,1,1,2,3,3,3,3,2,3,2,2,2,5,4,4,4,4,5,4,5,5,4,5]

    train_labels = [[i] for i in train_labels]
    train_labels = np.array(train_labels).astype(np.float32)
  
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    classes = []
    for test in scores:
        test = test.reshape(-1, 600).astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(test, k = 3)
        classes.append(int(result))
    print(classes)
    return classes

def drawScores(img, boxes, classes):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(0,0,255), (0,255,0), (255,0,0), (255, 255, 0), (0,255,255), (255,0,255)]
    names = ["wiolin", "basowy", "1", "1/2", "1/4", "1/8"]
    
    # 0 - klucz wiolinowy
    # 1 - klucz basowy
    # 2 - cala nuta
    # 3 - polnuta
    # 4 - cwierc
    # 5 - osemka

    for box, cl in zip(boxes, classes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), colors[cl], 2)
        cv2.putText(img, names[cl], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), thickness=2)
    saveAndShow("boxes.jpg", img)

    
######
img = getGrayImage("template2.jpg")
binary = binarize(img, blockSize=51, offset=10)
deskewed = deskew(binary)
lines = detectLines(deskewed)
imgWithLines = drawLines(deskewed, lines)
scores, boxes = splitScores(imgWithLines)
classes = classify(scores)
drawScores(deskew(img), boxes, classes)
######
