import logging
import score
import debug
from score.classes import Note


def __configure__logger__():
    import yaml
    from logging.config import dictConfig
    from pathlib import Path

    path = Path(__file__).parent.joinpath("logger.config.yaml").resolve()
    with open(path) as config_file:
        config = yaml.safe_load(config_file.read())
        dictConfig(config)


def get_notes(notes_rectangles, img):
    from score.classes import Note

    # Sort notes from left to right
    notes_rectangles.sort(key = lambda x: x[0])

    return [Note(x, y, w, h, img[y:y + h, x:x + w]) for x, y, w, h in notes_rectangles]


__configure__logger__()

_log = logging.getLogger('applogger')

file = input("Path to image: ")

img = score.io.import_image(file, (800, 500))
score.io.show_image('Start', img)

gray = score.filter.desaturate(img)
score.io.show_image('Desaturated', gray)
score.io.save_image('out/desaturated.png', gray)

binary = score.filter.binarize(gray)
score.io.show_image('Binarized', binary)
score.io.save_image('out/binarized.png', binary)

angle = score.transform.detect_rotation_angle(binary)
straight = score.transform.rotate(binary, angle)
score.io.show_image('Straightened', straight)
score.io.save_image('out/strightened.png', straight)

horizontal = score.filter.detect_horizontal_lines(binary)
score.io.show_image('Horizontal Lines', horizontal)
score.io.save_image('out/horizontal.png', horizontal)

vertical = score.filter.detect_vertical_lines(straight)
score.io.show_image('Vertical Lines', vertical)
score.io.save_image('out/vertical.png', vertical)

erased = score.filter.remove_lines(straight, horizontal, vertical)
score.io.show_image('Removed Lines', erased)
score.io.save_image('out/erased.png', erased)

notes_rectangles = score.filter.find_bounding_rectangles(binary, min_area = 150, max_area = 5000)
_log.debug(f"Found {len(notes_rectangles)} notes.")
rects = debug.__draw_rectangles__(binary, notes_rectangles)
score.io.show_image('Notes rectangles', rects)
# staff_lines_rectangles = find_bounding_rectangles(np.invert(horizontal_lines), min_area = 500, max_area = 400000)
staves = score.filter.find_staff_bounds(horizontal)
debug.__draw_staves__(horizontal, staves)

notes = get_notes(notes_rectangles, binary)

def classify(staves, notes: list[Note]):
    for note in notes:
        # Assign staff to every note.
        for staff in staves:
            if staff.contains(note.y_center):
                note.staff = staff
                break

        if note.staff is not None:
            note.classify()


classify(staves, notes)
final = score.io.overlay(img, notes, -angle)
score.io.show_image('Final', final)
score.io.save_image('out/final.png', final)

# ######
# import transform

# img = utils.import_image((800, 500))
# binary = filters.binarize(filters.desaturate(img), block_size = 51, offset = 10, filter = (9, 75, 75))

# angle = transform.detect_rotation_angle(binary)
# straight = transform.rotate(binary, angle)
# final = transform.rotate(img, angle)

# horizontal_lines = detect_horizontal_lines(straight)
# vertical_lines = detect_vertical_lines(straight)
# erased = remove_lines(straight, horizontal_lines, vertical_lines)

# notes_rectangles = find_bounding_rectangles(erased, min_area = 150, max_area = 5000)
# staff_lines_rectangles = find_bounding_rectangles(np.invert(horizontal_lines), min_area = 500, max_area = 400000)
# staves = find_staff_bounds(horizontal_lines)

# notes = get_notes(notes_rectangles, erased)

# classify(staves, notes)
# # draw_result(final, notes)
# utils.overlay(img, notes, -angle)
# ######
