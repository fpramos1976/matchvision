"""Montagem do quadro de saída e reprodução na velocidade real do vídeo.

A análise (YOLO + TrackNet) é mais lenta que o vídeo, principalmente em 4K
a 60 fps. Por isso o processamento grava o resultado anotado num arquivo e
a reprodução acontece depois, a partir desse arquivo, no ritmo original.
"""

import time

import cv2
import numpy as np

# Largura máxima do vídeo anotado (o mapa da quadra é acrescentado ao lado).
# Mantém o arquivo de saída leve e a reprodução fluida mesmo para vídeos 4K.
MAX_OUTPUT_WIDTH = 1600


def compose_frame(frame, bird_view, max_width=MAX_OUTPUT_WIDTH):
    """Junta o frame anotado (reduzido se for muito largo) e o mapa da
    quadra (na mesma altura) lado a lado."""
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA
        )
        height, width = frame.shape[:2]

    map_height, map_width = bird_view.shape[:2]
    map_scale = height / map_height
    bird_view = cv2.resize(bird_view, (int(round(map_width * map_scale)), height))

    composed = np.hstack([frame, bird_view])
    # Codecs de vídeo exigem dimensões pares
    even_h, even_w = composed.shape[0] // 2 * 2, composed.shape[1] // 2 * 2
    return composed[:even_h, :even_w]


def play_video(path, fps, window_name="MatchVision - Analise"):
    """Reproduz o vídeo anotado na velocidade original.

    Se a tela não acompanhar, pula quadros para não ficar em câmera lenta.
    Teclas: espaço pausa/continua, R reinicia, Q sai.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print(f"Erro: não foi possível abrir '{path}' para reprodução.")
        return

    frame_period = 1.0 / fps if fps > 0 else 1.0 / 30
    print("Reproduzindo na velocidade normal (espaço: pausa, R: reinicia, Q: sai).")

    start = time.perf_counter()
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            # Fim: mantém o último quadro na tela até R ou Q
            key = cv2.waitKey(0) & 0xFF
            if key == ord("r"):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start, index = time.perf_counter(), 0
                continue
            break

        due = start + index * frame_period
        now = time.perf_counter()
        if now > due + frame_period:
            # Atrasado: descarta este quadro para manter o ritmo real
            index += 1
            continue

        cv2.imshow(window_name, frame)
        wait_ms = max(1, int((due - time.perf_counter()) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        index += 1

        if key == ord("q"):
            break
        if key == ord(" "):
            paused_at = time.perf_counter()
            key = 0
            while key not in (ord(" "), ord("q")):
                key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            start += time.perf_counter() - paused_at
        elif key == ord("r"):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            start, index = time.perf_counter(), 0

    capture.release()
    cv2.destroyWindow(window_name)
