class Staff:
    def __init__(self, y, height, order):
        # self.x = x
        self.y = y
        # self.width = width
        self.height = height
        # self.staff_lines = staff_lines
        self.order = order
        # self.nparray = nparray
        self.y_center = y + (height // 2)
        # self.x_center = None # x + (width // 2)
        self.avg_staffline_space = height / 4

    def __str__(self) -> str:
        return f"(y: {self.y}, height: {self.height}, order: {self.order}, space between: {self.avg_staffline_space})"

    def contains(self, point) -> bool:
        return point >= self.y and point <= (self.y + self.height)

    def position(self, point) -> str:
        positions = ['1', '1-2', '2', '2-3', '3', '3-4', '4', '4-5', '5']
        r = self.avg_staffline_space / 2
        upper = self.y - r
        lower = self.y + r

        for position in positions:
            if point >= upper and point < lower:
                return position
            upper += r
            lower += r


class Note:
    class Head:
        def __init__(self, x, y, width, height, coverage) -> None:
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.coverage = coverage

    def __init__(self, x, y, width, height, nparray = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.nparray = nparray
        self.y_center = (2 * y + height) // 2
        self.x_center = (2 * x + width) // 2
        self.staff = None
        self.type = None
        self.position = None

    def __str__(self):
        # yapf: disable
        return (
            f"Nuta: {self.type:^7} | "
            f"Wysokość: {self.height:^5} | "
            f"Szerokość: {self.width:^5} | "
            f"Nr pięciolinii: {self.staff + 1:^5} | "
            f"Położenie {self.position:^7}"
        )
        # yapf: enable

    def classify(self):
        head = self._find_head()

        # If head takes all space or note's aspect ratio is landscape
        # then it's probably a whole note.
        if head.height == self.height or self.width >= self.height:
            self.type = '1'
        # If area covered by head is less than 50%
        # then it's probably a half note.
        elif head.coverage < 0.5:
            self.type = '1/2'
        # Otherwise we have to compare note's aspect ratios:
        # If aspect ratio is rather 'tall' (like <= 3 / 5) then
        # it's probably quarter note. Otherwise it's probably an eighth note.
        elif self.width / self.height < 0.6:
            self.type = '1/4'
        else:
            self.type = '1/8'

        self.position = self.staff.position(head.y + head.height * 0.8)

    def _find_head(self):
        import numpy as np
        half = self.height // 2
        upper_half = np.invert(self.nparray[:, :half]) / 255.
        lower_half = np.invert(self.nparray[:, half:]) / 255.

        # Calculate difference in image filling.
        # Half with more dense filling probably contains note's head.
        diff = np.sum(upper_half) - np.sum(lower_half)

        area = self.nparray.shape[0] * self.nparray.shape[1]
        area_half = area / 2
        diff_threshold = area / 8

        # If difference is smaller than given threshold
        # then note's head probably takes the whole space
        if abs(diff) < diff_threshold:
            return self.Head(self.x, self.y, self.width, self.height, np.sum(self.nparray) / area)
        # If difference is negative then note's head
        # is probably in lower half
        if diff < 0:
            return self.Head(self.x, self.y + half, self.width, half, np.sum(lower_half) / area_half)
        # If difference is positive then note's head
        # is probably in upper half
        if diff > 0:
            return self.Head(self.x, self.y, self.width, half, np.sum(upper_half) / area_half)

    def description(self) -> str:
        if self.type is not None:
            type = self.type
        else:
            type = "-"

        if self.staff is not None:
            staff = self.staff.order
        else:
            staff = "-"

        if self.position is not None:
            position = self.position
        else:
            position = "-"

        return f"[{staff}]({position})|{type}|"