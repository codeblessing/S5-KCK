# Ten moduł powinien się nazywać po prostu io, ale nie wiem czemu python ma problem z lokalnymi importami, więc tymczsowo jest `local io`.
import cv2 as cv
import config


def save_and_show(filename, img, path = config.OUTPUT_DIR):
    cv.imwrite(path + filename, img)
    cv.imshow(filename, img)
    cv.waitKey(0)