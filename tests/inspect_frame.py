"""
Ferramenta de debug: abre um frame específico do vídeo e permite
clicar num ponto (ex.: a bola) para imprimir suas coordenadas de
pixel exatas, além do tamanho real do frame (largura x altura).

Uso:
    python -m tests.inspect_frame videos/tennis_2.mp4 <numero_do_frame>

Clique no ponto desejado na janela que abrir. Pressione Q para sair.
"""

import sys
import cv2


def main() -> None:
    if len(sys.argv) < 3:
        print("Uso: python -m tests.inspect_frame <video_path> <frame_number>")
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

    height, width = frame.shape[:2]
    print(f"\nResolução real do frame: {width} x {height} (largura x altura)\n")
    print("Clique no ponto que quer medir (ex.: a bola). Pressione Q para sair.\n")

    display_frame = frame.copy()
    window_name = "Inspect Frame - clique no ponto de interesse"

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Ponto clicado: ({x}, {y})")
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, display_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()