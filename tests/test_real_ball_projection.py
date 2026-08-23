"""
Testa a projeção real de um ponto da bola, usando os 4 pontos de
calibração reais do vídeo tennis_2.mp4 e a posição real da bola
capturada com tests/inspect_frame.py.

Objetivo: descobrir se, mesmo que o ball_detector consiga detectar
a bola nessa posição, a Homography devolveria uma posição de mundo
plausível ou um valor absurdo (fora da quadra).
"""

from app.models.court_model import CourtModel
from app.models.court_corners import CourtCorners
from app.geometry.homography import Homography


def main() -> None:
    court = CourtModel(width=8.23, length=23.77)  # partida SINGLES, como no seu terminal

    corners = CourtCorners(
        top_left=(415, 254),
        top_right=(602, 278),
        bottom_right=(951, 496),
        bottom_left=(13, 377),
    )

    homography = Homography(image_points=corners, court_model=court)
    homography.compute()

    ball_pixel = (735, 44)
    world_x, world_y = homography.transform_point_to_world(ball_pixel)

    print(f"Bola no pixel: {ball_pixel}")
    print(f"Posição projetada no mundo: x={world_x:.2f} m, y={world_y:.2f} m")
    print(f"Quadra: width={court.width} m, length={court.length} m")

    margin_x, margin_y = 2.5, 5.0
    inside = (
        -margin_x <= world_x <= court.width + margin_x
        and -margin_y <= world_y <= court.length + margin_y
    )
    print(f"\nDentro da margem tolerada por _is_inside_court? {inside}")

    print("\n--- Simulação da nova lógica de _is_inside_court ---")
    calibrated_min_y = min(
        corners.top_left[1],
        corners.top_right[1],
        corners.bottom_right[1],
        corners.bottom_left[1],
    )
    print(f"Menor y calibrado: {calibrated_min_y}")
    print(f"Pixel da bola (y={ball_pixel[1]}) está acima do horizonte calibrado? "
          f"{ball_pixel[1] < calibrated_min_y}")


if __name__ == "__main__":
    main()