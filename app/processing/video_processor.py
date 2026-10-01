import os
import time
from pathlib import Path

import cv2

from app.calibration.manual_calibration import ManualCalibration
from app.detection.ball_detector import BallDetector
from app.detection.player_detector import PlayerDetector
from app.geometry.homography import Homography
from app.models.court_model import CourtModel
from app.models.court_type import CourtType
from app.utils.configuration import get_match_configuration
from app.utils.video_reader import open_video
from app.visualization.court_map import CourtMap
from app.visualization.playback import compose_frame, play_video


class VideoProcessor:

    def __init__(
        self,
        video_path: str,
        ball_detector: str | None = None,
        output_path: str | None = None,
        play: bool = True,
    ):
        self.video_path = video_path
        # Vídeo anotado gravado durante o processamento e reproduzido
        # depois na velocidade original.
        self.output_path = output_path or str(
            Path("output") / f"{Path(video_path).stem}_analise.mp4"
        )
        self.play = play
        # "tracknet" (padrão) ou "classico". Sem valor explícito, respeita
        # a variável de ambiente BALL_DETECTOR.
        self.ball_detector = (
            ball_detector or os.environ.get("BALL_DETECTOR") or "tracknet"
        ).lower()

    def _create_ball_detector(self, fps):
        """TrackNet por padrão (nos vídeos medidos acertou muito mais que o
        clássico). Se os pesos ou o PyTorch não estiverem disponíveis, cai
        para o detector clássico em vez de interromper a análise."""
        if self.ball_detector != "classico":
            try:
                from app.detection.tracknet_detector import TrackNetBallDetector

                detector = TrackNetBallDetector(fps=fps, max_buffer=25)
                print(f"Detector de bola: TrackNet ({detector.device})")
                return detector
            except Exception as error:
                print(
                    f"Aviso: TrackNet indisponível ({error}). "
                    "Usando o detector clássico."
                )

        print("Detector de bola: clássico (cor/movimento)")
        return BallDetector(max_buffer=25)

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
        max_players = 4 if configuration.court_type == CourtType.DOUBLES else 2
        player_detector = PlayerDetector(max_players=max_players)
        ball_detector = self._create_ball_detector(fps)

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
        # A análise é mais lenta que o vídeo (sobretudo em 4K/60 fps), então
        # cada quadro anotado é gravado em arquivo e a reprodução na
        # velocidade normal acontece no fim. Durante o processamento a
        # janela mostra uma prévia, sem esperar entre quadros.
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = None
        processed = 0
        started = time.perf_counter()
        interrupted = False

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # 1. Detecta e Rastreia os Jogadores
            players = player_detector.detect_and_track(
                frame, homography, court_model, fps=fps
            )

            # Extrai as Bounding Boxes dos jogadores com segurança
            player_boxes = [p["bbox"] for p in players if "bbox" in p]

            # 2. Detecta e Rastreia a Bola (ignorando ROI dos jogadores)
            ball_center, ball_world = ball_detector.detect(
                frame,
                homography=homography,
                court_model=court_model,
                player_boxes=player_boxes,
            )

            # 3. Desenha a quadra 2D (Bird's-Eye View) + HUD + Jogadores + Bola
            bird_view = court_map.draw_base_court()
            bird_view = court_map.draw_players(bird_view, players)
            bird_view = court_map.draw_ball(bird_view, ball_world)

            # 4. Desenha o rastro da bola no frame original
            frame = ball_detector.draw_ball_trail(frame)

            # Área onde o jogador do fundo é procurado (recorte ampliado)
            if player_detector.last_far_region is not None:
                rx1, ry1, rx2, ry2 = player_detector.last_far_region
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 1)

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

            # 6. Grava o quadro anotado (vídeo + mapa da quadra lado a lado)
            composed = compose_frame(frame, bird_view)
            if writer is None:
                height, width = composed.shape[:2]
                writer = cv2.VideoWriter(
                    self.output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(composed)
            processed += 1

            # Prévia e progresso (sem segurar o processamento)
            cv2.imshow("MatchVision - Processando", composed)
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            if total_frames:
                remaining = (total_frames - processed) / rate if rate > 0 else 0
                print(
                    f"\rProcessando: {processed}/{total_frames} quadros "
                    f"({100 * processed / total_frames:.0f}%) - "
                    f"{rate:.1f} quadros/s - faltam ~{remaining:.0f}s   ",
                    end="",
                    flush=True,
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nProcessamento interrompido pelo usuário.")
                interrupted = True
                break

        print()
        if writer is not None:
            writer.release()
            print(f"Vídeo analisado salvo em: {self.output_path}")
        cv2.destroyWindow("MatchVision - Processando")

        if self.play and writer is not None and not interrupted:
            play_video(self.output_path, fps)

        video.release()
        cv2.destroyAllWindows()
        print("MatchVision encerrado com sucesso.")