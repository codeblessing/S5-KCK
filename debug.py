import cv2 as cv


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


def __draw_staves__(img, staves):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for staff in staves:
        print(staff)
        img = cv.rectangle(img, (0, int(staff.y)), (1000, int(staff.y + staff.height)), (0, 255, 255), -1)
        img = cv.line(img, (0, int(staff.y)), (1000, int(staff.y)), (0, 0, 255), thickness = 2)

def __draw_rectangles__(img, rects):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for x, y, w, h in rects:
        img = cv.rectangle(img, (x, y, w, h), (0, 255, 255), -1)

    return img