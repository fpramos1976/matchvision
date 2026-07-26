import numpy as np

@dataclass
class DetectedLine:
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    length: float

def classify_lines(lines):
    horizontal = []
    vertical = []

    if lines is None:
        return horizontal, vertical

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = np.degrees(
            np.arctan2(y2 - y1, x2 - x1)
        )

        angle = abs(angle)

        if angle < 20:
            horizontal.append((x1, y1, x2, y2))

        elif angle > 70:
            vertical.append((x1, y1, x2, y2))

    return horizontal, vertical