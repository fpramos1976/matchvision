import cv2
import numpy as np


def detect_court_lines(frame):
    """
    Detecta as linhas da quadra.
    Atualmente usado apenas para depuração.
    """

    if frame is None:
        return None, None

    output_frame = frame.copy()
    height, width = frame.shape[:2]

    # Máscara para ignorar céu e árvores
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.rectangle(
        mask,
        (0, int(height * 0.42)),
        (width, int(height * 0.95)),
        255,
        -1,
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 70, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=40,
        maxLineGap=15,
    )

    if lines is None:
        return output_frame, edges

    for line in lines:

        x1, y1, x2, y2 = line.ravel()

        length = np.hypot(x2 - x1, y2 - y1)

        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        is_horizontal = angle < 5 or angle > 175
        is_diagonal = (
            (20 < angle < 65)
            or
            (115 < angle < 160)
        )

        if is_horizontal:

            cv2.line(
                output_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

        elif is_diagonal:

            if length < height * 0.5:

                cv2.line(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

    return output_frame, edges