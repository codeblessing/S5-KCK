import cv2 as cv
import numpy as np
import utils
import filters
import config
from datetime import datetime
from classes import Note, Staff
from pipe import select, where
import debug
import more_itertools as iter
import itertools


# Find horizontal lines (staff lines)
def detect_horizontal_lines(img):
    img = np.invert(img)

    horizontal_size = img.shape[1] // 30
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations = 1)

    if config.DEBUG:
        utils.save_and_show("horizontal.jpg", detected_lines)

    return detected_lines


# Find vertical lines (helpful in notes repairing)
def detect_vertical_lines(img):
    img = np.invert(img)

    vertical_size = img.shape[0] // 30
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations = 1)

    if config.DEBUG:
        utils.save_and_show("vertical.jpg", detected_lines)

    return detected_lines


# Remove horizontal lines and repair through vertical lines
def remove_lines(img, horizontal_lines, vertical_lines):
    img = cv.bitwise_or(img, horizontal_lines)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    img = np.invert(cv.bitwise_or(np.invert(img), vertical_lines))

    if config.DEBUG:
        utils.save_and_show("erased.jpg", img)

    return img


# Find rectangles of objects that have area between min_area and max_area
def find_bounding_rectangles(img, min_area, max_area):
    count, labels, stats = cv.connectedComponentsWithStats(np.invert(img))[:3]
    areas = stats[:, 4]

    for label in range(1, count):
        if areas[label] > max_area or areas[label] < min_area:
            labels[labels == label] = 0
    labels[labels > 0] = 255

    if config.DEBUG:
        time = datetime.now()
        labeled = labels.astype(np.uint8)
        utils.save_and_show(f'labeled_{time.hour}_{time.minute}_{time.second}_{time.microsecond}.png', labeled)

    stats = cv.connectedComponentsWithStats(labels.astype(np.uint8))[2]

    return [[x, y, w, h] for x, y, w, h, *_ in stats[1:]]


def find_staff_bounds(img):
    class Line:
        def __init__(self, dist, theta):
            self.dist = dist
            self.theta = theta
            self.y = np.round(np.sin(theta) * dist + 1000 * np.cos(theta))

        def __str__(self) -> str:
            return f"(r: {self.dist}, theta: {self.theta}, y: {self.y})"

    def find_approx_staff_height(lines):
        assert len(lines) > 2, "Error: cannot infer staff height from less than 2 lines."

        gradient = [lines[i].dist - lines[i - 1].dist for i in range(1, len(lines))]
        threshold = np.median(gradient) + np.std(gradient)
        spaces = list(enumerate(gradient) | where(lambda val: val[1] > threshold) | select(lambda val: val[0]))
        indices = sorted([i for i in itertools.chain(*[[i, i + 1] for i in spaces])] + [0, len(lines) - 1])
        heights = [lines[j].dist - lines[i].dist for i, j in iter.grouper(indices, 2, indices[-1])]

        return np.median(heights), indices

    # Find long horizontal lines (most probable staff lines).
    PI_HALF = np.pi / 2
    lines = cv.HoughLines(img, 1, np.pi / 360, img.shape[1] // 5)
    lines = list(
        sorted([Line(line[0][0], line[0][1]) for line in lines], key = lambda line: line.dist)
        | where(lambda line: abs(line.theta - PI_HALF) < 0.0001))

    print("lines:", *lines, sep = '\n')
    if config.DEBUG:
        debug.__draw_hough_lines__(img, lines, "staff_lines.png")
    height, indices = find_approx_staff_height(lines)

    staves = []
    for index in indices[::2]:
        staves.append(Staff(lines[index].y, height, index))

    if config.DEBUG:
        debug.__draw_staves__(img, staves)

    return staves


# def get_staves(staff_lines, img):
#     lines = []
#     staves = []

#     # Make sure that each staff consists of 5 lines
#     assert (len(lines) % 5 == 0)

#     # Sort lines from top to bottom
#     staff_lines.sort(key = lambda x: x[1])

#     # Create Staff objects
#     for index, line in enumerate(staff_lines):
#         x, y, w, h = line
#         lines.append(StaffLine(x, y, w, h, index % 5, img[y:y + h, x:x + w]))
#         if ((index + 1) % 5 == 0):
#             staff_height = (lines[-1].y + lines[-1].height) - lines[0].y
#             staff_width = max([line.width for line in lines])
#             nparray = img[lines[0].y:lines[0].y + staff_height, lines[0].x:lines[0].x + staff_width]
#             staves.append(Staff(lines[0].x, lines[0].y, staff_width, staff_height, lines, int(index / 5), nparray))
#             lines = []

#     return staves


def get_notes(notes_rectangles, img):
    notes = []

    # Sort notes from left to right
    notes_rectangles.sort(key = lambda x: x[0])

    # Create Note objects
    for note_rectangle in notes_rectangles:
        x, y, w, h = note_rectangle
        notes.append(Note(x, y, w, h, img[y:y + h, x:x + w]))

    return notes


def classify(staves, notes):
    # Find proper staff for each note (check distances between center of note and center of each staff)
    for note in notes:
        distances_from_staff = [(staff.order, abs(note.y_center - staff.y_center)) for staff in staves]
        distances_from_staff.sort(key = lambda x: x[1])
        note.staff = distances_from_staff[0][0]

    # Find proper note type for each note

    # First note on staff is always clef
    for i in range(len(staves)):
        # Common g-clef is always higher than staff
        if (notes[i].height > staves[notes[i].staff].height):
            notes[i].type = "wiolin"
        else:
            notes[i].type = "bas"

    # Other
    for note in notes:
        if (note.type is not None): continue
        assert staves[note.staff].order == note.staff

        half_height = int(note.height / 2)
        half_width = int(note.width / 2)

        # "1" has the size of space between staff lines
        if note.height < 1.1 * staves[note.staff].avg_staffline_space:
            note.type = "1"
        # "1/2" has many white pixels in bottom-left corner
        elif np.count_nonzero(note.nparray[half_height:, :half_width]) / (half_height * half_width) > 0.8:
            note.type = "1/2"
        # "1/8" has many white pixels in bottom-right corner
        elif np.count_nonzero(note.nparray[half_height:, half_width:]) / (half_height * half_width) > 0.8:
            note.type = "1/8"
        # Otherwise it's "1/4"
        else:
            note.type = "1/4"

    # Find position on staff
    for note in notes:
        # It's known where clef is located
        if (note.type == "bas" or note.type == "wiolin"):
            note.position = "-"
        # Otherwise I'm looking for nearest staff line (distances between certain point in note and center of line)
        else:
            # For "1" I take y-center, for others point that is lower (0.7)
            k = 0.5 if note.type == "1" else 0.7

            distances_from_line = [(line.order, abs((note.y + k * note.height) - line.y_center))
                                   for line in staves[note.staff].staff_lines]
            distances_from_line.sort(key = lambda x: x[1])
            line_nr, min = distances_from_line[0]

            # Note crosses the line
            if (min < 0.3 * staves[note.staff].avg_staffline_space):
                note.position = str(line_nr + 1)
            # Note is between lines
            else:
                note.position = "{}-{}".format(line_nr + 1, distances_from_line[1][0] + 1)

    # Check result
    flag_OK = True
    for note in notes:
        print(note)
        if (note.position is None or note.type is None or note.staff is None):
            flag_OK = False

    assert flag_OK


def classify(staves, notes):
    for note in notes:
        # Assign staff to every note.
        for staff in staves:
            if staff.contains(note.y_center):
                note.staff = staff
                break

        if note.staff is not None:
            note.classify()


######
import transform

img = utils.import_image((800, 500))
binary = filters.binarize(filters.desaturate(img), block_size = 51, offset = 10, filter = (9, 75, 75))

angle = transform.detect_rotation_angle(binary)
straight = transform.rotate(binary, angle)
final = transform.rotate(img, angle)

horizontal_lines = detect_horizontal_lines(straight)
vertical_lines = detect_vertical_lines(straight)
erased = remove_lines(straight, horizontal_lines, vertical_lines)

notes_rectangles = find_bounding_rectangles(erased, min_area = 150, max_area = 5000)
staff_lines_rectangles = find_bounding_rectangles(np.invert(horizontal_lines), min_area = 500, max_area = 400000)
staves = find_staff_bounds(horizontal_lines)

notes = get_notes(notes_rectangles, erased)

classify(staves, notes)
# draw_result(final, notes)
utils.overlay(img, notes, -angle)
######
