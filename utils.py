import sys
import config
import transform
import cv2 as cv
import numpy as np


# Shorthand for image saving and showing
def save_and_show(filename, img):
    path = config.OUTPUT_DIR
    cv.imwrite(path + filename, img)
    cv.imshow(filename, img)
    cv.waitKey(0)


def import_image(size):
    path = config.INPUT_DIR

    if (len(sys.argv) != 2):
        print("Usage: python3 main.py <filename>")
        print("where <filename> must be in {} directory".format(path))
        exit(0)

    img = cv.imread(path + sys.argv[1])
    if (img is None):
        print("Error. Cannot read '{}'".format(path + sys.argv[1]))
        exit(-1)

    return cv.resize(img, size)


def combine(overlay, background):
    if background.shape[2] == 1:
        background = cv.cvtColor(background, cv.COLOR_GRAY2RGB)

    assert overlay.shape[:
                         2] == background.shape[:
                                                2], f"Cannot combine images of diffent shapes!\nOverlay shape: {overlay.shape}\nBackground shape: {background.shape}"

    out = np.zeros(background.shape)

    alpha = overlay[:, :, 3] / 255.0
    out[:, :, 0] = (1. - alpha) * background[:, :, 0] + alpha * overlay[:, :, 0]
    out[:, :, 1] = (1. - alpha) * background[:, :, 1] + alpha * overlay[:, :, 1]
    out[:, :, 2] = (1. - alpha) * background[:, :, 2] + alpha * overlay[:, :, 2]

    return out


def overlay(img, notes, rotation):
    canvas = np.zeros((img.shape[0], img.shape[1], 4))

    for note in notes:
        # Text label example "wiolin 2 (3)" means "g-clef is placed on second staff and third line of this staff"
        text_label = note.description()

        cv.rectangle(canvas, (note.x, note.y), (note.x + note.width, note.y + note.height), (0, 255, 0, 255), 2)
        cv.putText(canvas,
                   text_label, (note.x, note.y),
                   cv.FONT_HERSHEY_COMPLEX,
                   0.3, (255, 0, 0, 255),
                   thickness = 1,
                   lineType = cv.LINE_AA)

    canvas = transform.rotate(canvas, rotation, cval = 0)
    out = combine(canvas, img)

    if config.DEBUG:
        save_and_show("result.jpg", out)

    return out