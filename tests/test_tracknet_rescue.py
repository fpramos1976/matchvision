"""Testes do resgate por cor do TrackNetBallDetector com quadros sintéticos.

Não rodam a rede: só a lógica que procura a bola amarela perto da posição
prevista quando o TrackNet não a vê.
"""

import os
import types

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from app.detection.tracknet_detector import TrackNetBallDetector

WEIGHTS = "models/tracknet.pt"
HEIGHT, WIDTH = 540, 960


def _bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


CLAY = _bgr(12, 110, 200)
BALL = _bgr(21, 190, 170)


def _frame(ball=None, radius=4):
    frame = np.full((HEIGHT, WIDTH, 3), CLAY, dtype=np.uint8)
    if ball is not None:
        cv2.circle(frame, ball, radius, BALL, -1)
    return frame


@pytest.fixture
def detector():
    if not os.path.exists(WEIGHTS):
        pytest.skip("pesos do TrackNet ausentes")
    det = TrackNetBallDetector(fps=24, device="cpu")
    # Trajetória em andamento: bola em (400, 300) indo 10 px/quadro para a direita
    det.last_center = (400, 300)
    det.velocity = (10.0, 0.0)
    det.missed_frames = 0
    return det


def _push(det, previous, current):
    det.frames.append(previous)
    det.frames.append(current)
    return current


def test_rescue_finds_moving_ball_near_prediction(detector):
    current = _push(detector, _frame((400, 300)), _frame((410, 300)))
    found = detector._rescue_by_color(current)
    assert found is not None
    assert abs(found[0] - 410) <= 2 and abs(found[1] - 300) <= 2


def test_rescue_ignores_static_yellow_patch(detector):
    # Capim seco / raquete parada: mesma mancha nos dois quadros
    current = _push(detector, _frame((410, 300)), _frame((410, 300)))
    assert detector._rescue_by_color(current) is None


def test_rescue_ignores_blob_inside_player_box(detector):
    current = _push(detector, _frame((400, 300)), _frame((410, 300)))
    assert detector._rescue_by_color(current, player_boxes=[(380, 250, 420, 350)]) is None


def test_rescue_keeps_ball_just_outside_player_box(detector):
    current = _push(detector, _frame((400, 300)), _frame((410, 300)))
    far_box = [(300, 250, 380, 350)]  # termina 30 px antes da bola
    assert detector._rescue_by_color(current, player_boxes=far_box) is not None


def test_rescue_never_starts_a_trajectory(detector):
    detector.last_center = None
    current = _push(detector, _frame((400, 300)), _frame((410, 300)))
    assert detector._rescue_by_color(current) is None


def test_rescue_gives_up_after_window_and_streak(detector):
    current = _push(detector, _frame((400, 300)), _frame((410, 300)))
    detector.missed_frames = detector.rescue_window
    assert detector._rescue_by_color(current) is None

    detector.missed_frames = 0
    detector.rescue_streak = detector.max_rescue_streak
    assert detector._rescue_by_color(current) is None


def test_rescue_ignores_ball_far_from_prediction(detector):
    current = _push(detector, _frame((100, 100)), _frame((110, 100)))
    assert detector._rescue_by_color(current) is None


def test_detect_uses_rescue_when_network_sees_nothing(detector):
    detector._heatmap = types.MethodType(
        lambda self: np.zeros((detector.INPUT_HEIGHT, detector.INPUT_WIDTH), np.uint8), detector
    )
    detector.frames.append(_frame((390, 300)))
    detector.frames.append(_frame((400, 300)))
    center, _ = detector.detect(_frame((410, 300)))
    assert center is not None and abs(center[0] - 410) <= 2
    assert detector.rescue_streak == 1 and detector.missed_frames == 0


def test_color_rescue_can_be_disabled():
    if not os.path.exists(WEIGHTS):
        pytest.skip("pesos do TrackNet ausentes")
    det = TrackNetBallDetector(fps=24, device="cpu", color_rescue=False)
    det._heatmap = types.MethodType(
        lambda self: np.zeros((det.INPUT_HEIGHT, det.INPUT_WIDTH), np.uint8), det
    )
    det.last_center = (400, 300)
    det.frames.append(_frame((390, 300)))
    det.frames.append(_frame((400, 300)))
    center, _ = det.detect(_frame((410, 300)))
    assert center is None
