"""Preenchimento de buracos curtos do TrackNetBallDetector, sem carregar a
rede: a posição "vista" pela rede em cada frame é injetada diretamente."""

import collections

import numpy as np

from app.detection.tracknet_detector import TrackNetBallDetector


def make_detector(fps, seen):
    detector = object.__new__(TrackNetBallDetector)
    detector.frame_step = 1
    detector.frames = collections.deque(maxlen=3)
    detector.heatmap_threshold = 127
    detector.max_jump = 100
    detector.fps = fps
    detector._pending_fill = None
    detector.max_gap = max(1, int(round(fps / 6.0)))
    detector.trajectory_pixels = collections.deque(maxlen=60)
    detector.trajectory_world = collections.deque(maxlen=60)
    detector.last_center = None
    detector.missed_frames = 0
    # Estado do resgate por cor (desligado: estes testes cobrem só o
    # preenchimento de buracos a partir do que a rede vê)
    detector.color_rescue = False
    detector.velocity = (0.0, 0.0)
    detector.rescue_streak = 0
    detector._recent = collections.deque(
        maxlen=max(6, int(round(fps * TrackNetBallDetector.STATIONARY_SECONDS)))
    )
    detector._heatmap = lambda: None
    spots = iter(seen)
    # Coordenadas injetadas já em pixels de 640x360 (frame de mesmo tamanho)
    detector._ball_from_heatmap = lambda _: next(spots)
    return detector


def run(fps, seen):
    detector = make_detector(fps, seen)
    frame = np.zeros((360, 640, 3), np.uint8)
    for _ in range(2):
        detector.frames.append(frame)
    outputs = [detector.detect(frame)[0] for _ in seen]
    return outputs, list(reversed(detector.trajectory_pixels))


def test_short_gap_is_filled_in_trail():
    seen = [(100 + 10 * i, 200) if i not in (4, 5, 6) else None for i in range(10)]
    outputs, trail = run(60, seen)
    # Saída ao vivo não inventa posições
    assert outputs[4:7] == [None, None, None]
    # O rastro é completado em linha reta entre as detecções vizinhas
    assert trail[4:7] == [(140, 200), (150, 200), (160, 200)]


def test_long_gap_is_not_filled():
    seen = [(100 + 10 * i, 200) if not 2 <= i <= 14 else None for i in range(18)]
    _, trail = run(60, seen)  # max_gap = 10 a 60 fps
    assert all(p is None for p in trail[2:15])


def test_gap_after_jump_is_not_filled():
    seen = [(100, 200), (110, 200), None, (600, 50)]
    _, trail = run(60, seen)
    assert trail[2] is None


def test_ball_rolling_on_ground_is_hidden_after_one_second():
    # Bola fora de jogo rolando ~2,5 px/quadro num vídeo de 1920 px (como
    # no tennis_2.mp4); o quadro do teste tem 640 px, daí ~0,8 px/quadro
    seen = [(int(round(400 - 0.8 * i)), 300) for i in range(40)]
    outputs, _ = run(24, seen)
    window = int(round(24 * TrackNetBallDetector.STATIONARY_SECONDS))
    assert all(o is not None for o in outputs[: window - 1])
    assert all(o is None for o in outputs[window:])


def test_slow_ball_near_top_of_lob_is_kept():
    # Bola em jogo que desacelera por ~0,4 s no topo da trajetória (em
    # pixels do quadro de 640 px do teste)
    xs, x = [], 100.0
    for i in range(50):
        x += 1 if 20 <= i < 30 else 6
        xs.append(int(x))
    seen = [(x, 200) for x in xs]
    outputs, _ = run(24, seen)
    assert all(o is not None for o in outputs)


def test_jump_limit_follows_fps():
    # Quadro de 640 px: 35 px/quadro = 105 px num vídeo de 1920 px.
    # A 24 fps isso é ~2.500 px/s (bola rápida plausível); a 60 fps,
    # ~6.300 px/s, rápido demais para ser a mesma bola.
    seen = [(100 + 10 * i, 200) for i in range(5)] + [(175, 200)]
    out_24, _ = run(24, seen)
    out_60, _ = run(60, seen)
    assert out_24[-1] == (175, 200)
    assert out_60[-1] is None


def test_isolated_spike_is_removed_from_trail():
    seen = [(100, 200), (110, 200), (140, 170), (130, 200), (140, 200)]
    outputs, trail = run(24, seen)
    assert outputs[2] == (140, 170)  # ao vivo ele foi mostrado
    assert trail[2] is None          # mas sai do rastro
    assert trail[3] == (130, 200)


def test_far_point_after_gap_needs_confirmation_before_filling():
    # Bola some 4 quadros e "reaparece" longe (ponto falso isolado): o
    # rastro não pode ligar os dois em linha reta
    base = [(100 + 5 * i, 200) for i in range(4)]
    isolated = base + [None] * 4 + [(200, 120), None, None]
    _, trail = run(24, isolated)
    assert all(p is None for p in trail[4:8])

    # Mesmo salto, mas confirmado por uma detecção no quadro seguinte
    confirmed = base + [None] * 4 + [(200, 120), (205, 116)]
    _, trail = run(24, confirmed)
    assert all(p is not None for p in trail[4:8])
