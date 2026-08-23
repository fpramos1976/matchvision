"""
Testes sintéticos de geometria para o MatchVision.

Objetivo:
1. Validar que a Homography faz corretamente o round-trip
   mundo -> imagem -> mundo para os 4 cantos da quadra.
2. Quantificar o erro que a Homography introduz quando o ponto
   observado (ex.: a bola) NÃO está no plano do solo (Z > 0),
   provando numericamente por que a bola aparece deslocada no
   Bird's-Eye View.

Não depende de nenhum vídeo real: usamos uma câmera sintética
(pinhole) com posição e orientação conhecidas para gerar os
pontos de imagem, então comparamos com o que a Homography
"acha" que é a posição no mundo.
"""

import numpy as np
import cv2

from app.models.court_model import CourtModel
from app.models.court_corners import CourtCorners
from app.geometry.homography import Homography


def look_at_rotation(camera_pos, target_pos, world_up=(0.0, 0.0, 1.0)):
    """Calcula a matriz de rotação mundo->câmera (convenção OpenCV:
    X direita, Y para baixo, Z para frente)."""
    camera_pos = np.array(camera_pos, dtype=np.float64)
    target_pos = np.array(target_pos, dtype=np.float64)
    world_up = np.array(world_up, dtype=np.float64)

    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    rotation = np.vstack([right, -true_up, forward])
    return rotation


def build_synthetic_camera(camera_pos, target_pos, focal_length=800.0,
                            image_size=(1280, 720)):
    """Retorna (K, rvec, tvec) de uma câmera sintética apontando para target_pos."""
    width_px, height_px = image_size
    rotation = look_at_rotation(camera_pos, target_pos)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ np.array(camera_pos, dtype=np.float64)

    k_matrix = np.array([
        [focal_length, 0.0, width_px / 2.0],
        [0.0, focal_length, height_px / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return k_matrix, rvec, tvec.reshape(3, 1)


def project_point(world_point_3d, k_matrix, rvec, tvec):
    obj_points = np.array([world_point_3d], dtype=np.float64)
    image_points, _ = cv2.projectPoints(obj_points, rvec, tvec, k_matrix, None)
    x, y = image_points[0, 0]
    return float(x), float(y)


def build_calibrated_homography(court, k_matrix, rvec, tvec):
    """Simula o clique manual dos 4 cantos, projetando os cantos reais
    da quadra (Z=0) através da câmera sintética."""

    top_left_world = (0.0, 0.0, 0.0)
    top_right_world = (court.width, 0.0, 0.0)
    bottom_right_world = (court.width, court.length, 0.0)
    bottom_left_world = (0.0, court.length, 0.0)

    corners = CourtCorners(
        top_left=project_point(top_left_world, k_matrix, rvec, tvec),
        top_right=project_point(top_right_world, k_matrix, rvec, tvec),
        bottom_right=project_point(bottom_right_world, k_matrix, rvec, tvec),
        bottom_left=project_point(bottom_left_world, k_matrix, rvec, tvec),
    )

    homography = Homography(image_points=corners, court_model=court)
    homography.compute()
    return homography


def test_ground_corners_round_trip():
    """Os 4 cantos da quadra (Z=0) devem voltar exatamente aos valores
    originais depois de imagem -> mundo. Isso valida a ordem dos pontos
    e o cálculo da Homography."""

    court = CourtModel()  # duplas: 10.97 x 23.77
    camera_pos = (court.width / 2.0, -8.0, 3.0)
    target_pos = (court.width / 2.0, court.length / 2.0, 0.0)

    k_matrix, rvec, tvec = build_synthetic_camera(camera_pos, target_pos)
    homography = build_calibrated_homography(court, k_matrix, rvec, tvec)

    expected_world_points = [
        (0.0, 0.0),
        (court.width, 0.0),
        (court.width, court.length),
        (0.0, court.length),
    ]

    image_points_in_order = [
        homography.image_points.top_left,
        homography.image_points.top_right,
        homography.image_points.bottom_right,
        homography.image_points.bottom_left,
    ]

    for (expected_x, expected_y), image_point in zip(
        expected_world_points, image_points_in_order
    ):
        recovered_x, recovered_y = homography.transform_point_to_world(image_point)
        assert abs(recovered_x - expected_x) < 1e-3, (
            f"Esperado x={expected_x}, obtido {recovered_x}"
        )
        assert abs(recovered_y - expected_y) < 1e-3, (
            f"Esperado y={expected_y}, obtido {recovered_y}"
        )

    print("OK: os 4 cantos fazem round-trip mundo->imagem->mundo corretamente.")


def test_ball_height_causes_projection_error():
    """Demonstra que, quando o ponto observado tem altura Z > 0 (como a
    bola no ar), a Homography (que assume Z=0) devolve uma posição no
    mundo diferente da posição real (x, y) da bola no chão, e o erro
    cresce com a altura."""

    court = CourtModel()
    camera_pos = (court.width / 2.0, -8.0, 3.0)
    target_pos = (court.width / 2.0, court.length / 2.0, 0.0)

    k_matrix, rvec, tvec = build_synthetic_camera(camera_pos, target_pos)
    homography = build_calibrated_homography(court, k_matrix, rvec, tvec)

    true_ground_x, true_ground_y = court.width / 2.0, 10.0

    print(f"\nPosição real da bola no chão: ({true_ground_x:.2f}, {true_ground_y:.2f}) m")
    print("Altura(m) | Ponto na imagem       | Mundo recuperado (Z=0)   | Erro (m)")

    previous_error = -1.0
    for ball_height in (0.0, 0.3, 0.8, 1.5, 2.5):
        ball_world_3d = (true_ground_x, true_ground_y, ball_height)
        image_point = project_point(ball_world_3d, k_matrix, rvec, tvec)

        recovered_x, recovered_y = homography.transform_point_to_world(image_point)
        error = float(np.hypot(recovered_x - true_ground_x, recovered_y - true_ground_y))

        print(
            f"{ball_height:8.2f}  | ({image_point[0]:7.1f}, {image_point[1]:6.1f}) "
            f"| ({recovered_x:6.2f}, {recovered_y:6.2f})     | {error:6.2f}"
        )

        if ball_height == 0.0:
            assert error < 1e-2, "Com altura 0 o erro deveria ser ~zero."
        else:
            assert error > previous_error, (
                "Esperava-se erro crescente com a altura; "
                "se isso falhar, revise a câmera sintética."
            )
        previous_error = error


if __name__ == "__main__":
    test_ground_corners_round_trip()
    test_ball_height_causes_projection_error()