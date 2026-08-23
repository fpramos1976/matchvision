"""
Ferramenta de debug: abre um frame específico do vídeo e, ao clicar
num ponto (idealmente o centro da bola), imprime o valor HSV real
daquele pixel. Usado para calibrar com precisão a faixa de cor usada
em BallDetector._create_yellow_mask, em vez de estimar valores.

Uso:
    python -m tests.inspect_ball_color videos/tennis_2.mp4 <numero_do_frame>
"""

import sys
import cv2


def main() -> None:
    if len(sys.argv) < 3:
        print("Uso: python -m tests.inspect_ball_color <video_path> <frame_number>")
        sys.exit(1)

    video_path = sys.argv[1]
    target_frame_number = int(sys.argv[2])

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Erro: não foi possível abrir '{video_path}'.")
        sys.exit(1)

    frame = None
    for _ in range(target_frame_number + 1):
        ok, frame = capture.read()
        if not ok:
            print("Erro: acabou o vídeo antes de chegar no frame pedido.")
            sys.exit(1)

    capture.release()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    display_frame = frame.copy()
    window_name = "Inspect Ball Color - clique na bola, Q para sair"

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv_frame[y, x]
            print(f"Pixel ({x}, {y}) -> HSV: H={h} S={s} V={v}")
            cv2.circle(display_frame, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow(window_name, display_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    while True:
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()