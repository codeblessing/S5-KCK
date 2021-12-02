import cv2 as cv
import utils


def __draw_hough_lines__(img, lines, filename = "detected_staff_lines.png"):
    import math

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for line in lines:
        a = math.cos(line.theta)
        b = math.sin(line.theta)
        x0 = a * line.dist
        y0 = b * line.dist
        start = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        end = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(img, start, end, (0, 0, 255), 1, cv.LINE_AA)
    utils.save_and_show(filename, img)

def __draw_staves__(img, staves):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for staff in staves:
        print(staff)
        img = cv.rectangle(img, (0, int(staff.y)), (1000, int(staff.y + staff.height)), (255, 255, 0), -1)

    utils.save_and_show('staves.png', img)