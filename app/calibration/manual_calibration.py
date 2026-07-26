import cv2


def start_calibration(frame):
    """Inicia o processo de calibração manual da quadra."""

    print("Iniciando calibração manual...")

    cv2.imshow("Manual Calibration", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyWindow("Manual Calibration")