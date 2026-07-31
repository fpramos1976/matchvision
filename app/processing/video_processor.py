import cv2

from app.calibration.manual_calibration import ManualCalibration
from app.detection.ball_detector import BallDetector
from app.detection.player_detector import PlayerDetector
from app.geometry.homography import Homography
from app.models.court_model import CourtModel
from app.utils.configuration import get_match_configuration
from app.utils.video_reader import open_video
from app.visualization.court_map import CourtMap


class VideoProcessor:

    def __init__(self, video_path: str):
        self.video_path = video_path

    def run(self):
        print("Hello from MatchVision!")

        # ==========================================
        # Configuração da partida
        # ==========================================
        configuration = get_match_configuration()

        # ==========================================
        # Abre o vídeo para Calibração
        # ==========================================
        video = open_video(self.video_path)
        if video is None:
            print("Encerrando o MatchVision.")
            return

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        frame_delay = max(1, int(1000 / fps))

        # Primeiro frame para calibração manual
        ret, frame = video.read()
        if not ret:
            print("Erro ao ler o primeiro frame.")
            video.release()
            return

        # ==========================================
        # Calibração & Homografia
        # ==========================================
        calibration = ManualCalibration(frame, configuration)
        points = calibration.run()

        court_model = CourtModel(
            width=configuration.court_type.value["width"],
            length=configuration.court_type.value["length"],
        )

        court_map = CourtMap(court_model=court_model)

        homography = Homography(
            image_points=points,
            court_model=court_model,
        )
        homography.compute()

        # ==========================================
        # Instancia os Detectores (Jogadores e Bola)
        # ==========================================
        player_detector = PlayerDetector(confidence=0.5)
        ball_detector = BallDetector(max_buffer=25)

        # ==========================================
        # Reabre o vídeo para o Loop
        # ==========================================
        video.release()
        video = open_video(self.video_path)

        if video is None:
            print("Erro ao reabrir o vídeo.")
            return

        # ==========================================
        # Loop do Processamento de Vídeo
        # ==========================================
        while True:
            ret, frame = video.read()
            if not ret:
                print("Fim do vídeo.")
                break

            # 1. Detecta e Rastreia os Jogadores
            players = player_detector.detect_and_track(
                frame, homography, court_model, fps=fps
            )

            # Extrai as Bounding Boxes dos jogadores com segurança
            player_boxes = [p["bbox"] for p in players if "bbox" in p]

            # 2. Detecta e Rastreia a Bola (ignorando ROI dos jogadores + filtro físico)
            ball_center, ball_world = ball_detector.detect(
                frame, homography, court_model, player_boxes=player_boxes
            )

            # 3. Desenha a quadra 2D (Bird's-Eye View) + HUD + Jogadores + Bola
            bird_view = court_map.draw_base_court()
            bird_view = court_map.draw_players(bird_view, players)
            bird_view = court_map.draw_ball(bird_view, ball_world)

            # 4. Desenha o rastro da bola no frame original
            frame = ball_detector.draw_ball_trail(frame)

            # 5. Desenha as caixas e métricas dos jogadores sobre o frame original
            for p in players:
                x1, y1, x2, y2 = p["bbox"]
                wx, wy = p["world_pos"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                info_str = f"P{p['id']} ({wx:.1f}m, {wy:.1f}m) - {p['speed']:.1f}km/h"
                cv2.putText(
                    frame,
                    info_str,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Janelas de exibição
            cv2.imshow("Original", frame)
            cv2.imshow("Bird's-Eye View (SwingVision)", bird_view)

            tecla = cv2.waitKey(frame_delay) & 0xFF
            if tecla == ord("q"):
                print("Processamento interrompido pelo usuário.")
                break

        video.release()
        cv2.destroyAllWindows()
        print("MatchVision encerrado com sucesso.")