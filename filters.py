import cv2 as cv
import utils
import config

def desaturate(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.fastNlMeansDenoising(gray, None)

    if config.DEBUG:
        utils.save_and_show("gray.jpg", gray)

    return gray

def binarize(img, block_size, offset, filter):
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, offset)
    thresh = cv.bilateralFilter(thresh, *filter)

    if config.DEBUG:
        utils.save_and_show("thresh.jpg", thresh)
        
    return thresh