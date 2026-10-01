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
    detector.max_gap = max(1, int(round(fps / 6.0)))
    detector.trajectory_pixels = collections.deque(maxlen=60)
    detector.trajectory_world = collections.deque(maxlen=60)
    detector.last_center = None
    detector.missed_frames = 0
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
