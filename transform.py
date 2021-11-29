# External modules
import numpy as np
import cv2 as cv
import scipy.ndimage.interpolation as interpolation


def detect_rotation_angle(img):
    # Hough Transform takes black pixels as background, so we need to invert image.
    inv = np.invert(img)
    lines: list = [(line[0][0], line[0][1]) for line in cv.HoughLines(inv, 1, np.pi / 360.0, int(img.shape[0] / 1.5))]
    angles = [line[1] for line in lines]
    median = np.median(angles)

    # Due to numerical errors angles are rounded to two decimal points.
    return round(median * 180 / np.pi - 90.0, 2)


def rotate(img, angle, cval = 255, reshape = False):
    return interpolation.rotate(img, angle, reshape = reshape, order = 0, cval = cval)
