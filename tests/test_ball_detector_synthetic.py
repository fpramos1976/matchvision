"""
Testes sintéticos do BallDetector (sem vídeo real).

Gera frames de uma "quadra" verde com uma bola amarela percorrendo uma
trajetória conhecida e mede quantos frames o detector encontra a bola e
com que erro em pixels.
"""

import numpy as np
import cv2

from app.detection.ball_detector import BallDetector

WIDTH, HEIGHT = 640, 480
# Cor da bola em BGR a partir de HSV medido no vídeo real (H~18, S~120, V~170)
BALL_BGR = tuple(
    int(c)
    for c in cv2.cvtColor(np.uint8([[[18, 120, 170]]]), cv2.COLOR_HSV2BGR)[0][0]
)
COURT_BGR = (60, 110, 60)
PLAYER_BGR = (200, 200, 200)


def make_frame(ball_pos, player_box=None, rng=None):
    frame = np.full((HEIGHT, WIDTH, 3), COURT_BGR, dtype=np.uint8)
    if rng is not None:
        noise = rng.integers(-3, 4, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if player_box is not None:
        x1, y1, x2, y2 = player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_BGR, -1)
    if ball_pos is not None:
        cv2.circle(frame, ball_pos, 3, BALL_BGR, -1, cv2.LINE_AA)
    return frame


def run(positions, player_boxes=None):
    detector = BallDetector(show_debug=False, verbose_logging=False)
    rng = np.random.default_rng(0)
    results = []
    for i, pos in enumerate(positions):
        box = player_boxes[i] if player_boxes else None
        frame = make_frame(pos, box, rng)
        center, _ = detector.detect(
            frame, player_boxes=[box] if box is not None else None
        )
        results.append(center)
    return results


def linear_path(n=40, start=(100, 350), velocity=(12, -5)):
    return [
        (int(start[0] + velocity[0] * i), int(start[1] + velocity[1] * i))
        for i in range(n)
    ]


def test_moving_ball_is_tracked_accurately():
    positions = linear_path()
    results = run(positions)

    errors = [
        np.hypot(r[0] - p[0], r[1] - p[1])
        for r, p in zip(results[3:], positions[3:])
        if r is not None
    ]
    detected = len(errors)

    assert detected >= 0.9 * (len(positions) - 3)
    # Sem "fantasmas" de frames anteriores: a detecção fica na posição atual
    assert np.median(errors) <= 2.0


def test_ball_tracked_while_passing_next_to_player():
    positions = linear_path()
    # Jogador parado cuja caixa é atravessada pela bola entre os frames 20-26
    box = (330, 180, 400, 300)
    results = run(positions, player_boxes=[box] * len(positions))

    inside = [
        i
        for i, (x, y) in enumerate(positions)
        if box[0] - 5 <= x <= box[2] + 5 and box[1] - 5 <= y <= box[3] + 5
    ]
    assert inside, "a trajetória de teste deveria cruzar a caixa do jogador"

    hits = [
        i
        for i in inside
        if results[i] is not None
        and np.hypot(
            results[i][0] - positions[i][0], results[i][1] - positions[i][1]
        )
        <= 4.0
    ]
    assert len(hits) >= len(inside) // 2


def test_static_yellow_blob_is_not_detected():
    # Objeto amarelo parado na altura da rede (ex.: ilhó) não pode virar bola
    positions = [(320, 300)] * 30
    results = run(positions)
    assert sum(r is not None for r in results) <= 3


def test_camera_shake_does_not_track_the_net():
    """Câmera tremendo 1-2 px: as bordas da rede/alambrado não podem virar
    'movimento' e roubar o lugar da bola."""
    rng = np.random.default_rng(3)
    big = np.full((HEIGHT + 40, WIDTH + 40, 3), COURT_BGR, dtype=np.uint8)
    big[:150] = (200, 170, 120)
    for _ in range(60):
        x, y = rng.integers(0, WIDTH + 40), rng.integers(0, 150)
        color = tuple(int(v) for v in rng.integers(40, 230, 3))
        cv2.rectangle(big, (x, y), (x + rng.integers(10, 60), y + rng.integers(10, 60)), color, -1)
    for x in range(0, WIDTH + 40, 12):
        cv2.line(big, (x, 150), (x, 200), (150, 150, 150), 1)
    cv2.line(big, (60, 215), (WIDTH - 20, 215), (250, 250, 250), 3)

    positions = linear_path(n=50, start=(80, 340), velocity=(9, -4))
    detector = BallDetector(show_debug=False, verbose_logging=False)
    correct = wrong = 0
    for pos in positions:
        dx, dy = rng.uniform(-2, 2, 2)
        matrix = np.float32([[1, 0, -20 + dx], [0, 1, -20 + dy]])
        frame = big.copy()
        cv2.circle(frame, (pos[0] + 20, pos[1] + 20), 3, BALL_BGR, -1, cv2.LINE_AA)
        frame = cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
        center, _ = detector.detect(frame)
        if center is None:
            continue
        true_x, true_y = pos[0] + dx, pos[1] + dy
        if np.hypot(center[0] - true_x, center[1] - true_y) <= 5:
            correct += 1
        else:
            wrong += 1

    assert correct >= 40
    assert wrong <= 3
