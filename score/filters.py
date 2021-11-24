import cv2 as cv
import numpy as np

from pipe import where, map
from score import io, config


class Line:
    def __init__(self, r, theta) -> None:
        self.r = r
        self.theta = theta

    def __str__(self) -> str:
        return f"(r: {self.r}, θ: {self.theta})"


def desaturate(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.fastNlMeansDenoising(gray)

    if config.DEBUG:
        io.save_and_show("gray.jpg", gray)

    return gray


def binarize(img, block_size, offset):
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, offset)
    thresh = cv.bilateralFilter(thresh, 9, 75, 75)

    if config.DEBUG:
        io.save_and_show("thresh.jpg", thresh)

    return thresh


def detect_rotation_angle(img):
    # Hough Transform takes black pixels as background, so we need to invert image.
    inv = np.invert(img)
    lines: list = [
        Line(line[0][0], line[0][1]) for line in cv.HoughLines(inv, 1, np.pi / 360.0, int(img.shape[0] / 1.5))
    ]
    angles = [line.theta for line in lines]
    median = np.median(angles)

    if config.DEBUG:
        stddev = np.std(angles)
        lines = list(lines | where(lambda line: abs(line.theta - median) <= stddev))
        mean = np.mean([line.theta for line in lines])
        lines = list(lines | map(lambda line: Line(line.r, median)))
        print("Mean:", mean)
        print("Median:", median)
        __draw_hough_lines__(img, lines)

    # Due to numerical errors angles are rounded to two decimal points.
    return round(median * 180 / np.pi - 90.0, 2)


def remove_horizontal_lines(img):
    out = []
    for row in img:
        if sum(row) > img.shape[0] * 255:
            out.append(row)

    if config.DEBUG:
        io.save_and_show("erased_lines.png", np.array(out))

    return np.array(out)


### DEBUG ONLY ###
def __draw_hough_lines__(img, lines, filename = "detected_staff_lines.png"):
    import math

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for line in lines:
        a = math.cos(line.theta)
        b = math.sin(line.theta)
        x0 = a * line.r
        y0 = b * line.r
        start = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        end = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(img, start, end, (0, 0, 255), 3, cv.LINE_AA)
    io.save_and_show(filename, img)


### DEBUG ONLY ###