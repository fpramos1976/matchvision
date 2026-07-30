import cv2

from app.utils.video_reader import open_video
from app.utils.configuration import get_match_configuration

from app.calibration.manual_calibration import ManualCalibration
from app.geometry.homography import Homography
from app.models.court_model import CourtModel

# Temporário (iremos remover quando a Bird's-Eye View estiver pronta)
from app.calibration.court_detector import detect_court_lines
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

        print(f"Tipo da partida: {configuration.court_type.value}")

        # ==========================================
        # Abre o vídeo
        # ==========================================
        video = open_video(self.video_path)

        if video is None:
            print("Encerrando o MatchVision.")
            return

        print(f"Total de frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"FPS: {video.get(cv2.CAP_PROP_FPS):.2f}")
        fps = video.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        frame_delay = max(1, int(1000 / fps))

        print(f"Tempo entre frames: {frame_delay} ms")

        # ==========================================
        # Primeiro frame (calibração)
        # ==========================================
        ret, frame = video.read()

        if not ret:
            print("Erro ao ler o primeiro frame.")
            video.release()
            return

        print("Pronto para começar a processar os frames!")

        # ==========================================
        # Calibração Manual
        # ==========================================
        calibration = ManualCalibration(frame, configuration)

        points = calibration.run()

        print()
        print(f"Pontos selecionados: {points}")

        # ==========================================
        # Homografia
        # ==========================================
        # ==========================================
        # Homografia
        # ==========================================
        court_model = CourtModel(
            width=configuration.court_type.value["width"],
            length=configuration.court_type.value["length"],
        )

        court_map = CourtMap(
            court_model=court_model,
        )


        homography = Homography(
            image_points=points,
            court_model=court_model,
        )

        H = homography.compute()

        print("Homografia calculada!")
        print("Matriz da homografia:")
        print(H)

        # ==========================================
        # Reinicia o vídeo
        # ==========================================
        # Fecha o vídeo usado apenas para calibração
        video.release()

        # Reabre o vídeo para iniciar o processamento
        video = open_video(self.video_path)

        if video is None:
            print("Erro ao reabrir o vídeo.")
            return

        print("Vídeo reaberto para processamento.")
        print(f"Frame atual: {video.get(cv2.CAP_PROP_POS_FRAMES)}")
        print(f"Total de frames: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")

        # ==========================================
        # Processamento
        # ==========================================
        contador = 0

        while True:

            contador += 1

            ret, frame = video.read()
            print(f"read() retornou: {ret}")

            if not ret:
                print("Fim do vídeo.")
                break

            # Bird's-Eye View (por enquanto devolve o frame original)
            bird_view = court_map.create()

            # ==========================================
            # Teste da Homografia - Centro da quadra
            # ==========================================

            court_center = (
                court_model.width / 2,
                court_model.length / 2,
            )

            center_pixel = homography.transform_point(court_center)

            cv2.circle(
                frame,
                center_pixel,
                10,
                (0, 0, 255),
                -1,
            )

            print(f"Centro da quadra: {court_center}")
            print(f"Pixel projetado: {center_pixel}")

            # Detector antigo (apenas para comparação)
            frame_com_linhas, edges = detect_court_lines(frame)
            print(f"Frame {contador}")


            # Janelas
            cv2.imshow("Original", frame)

            if bird_view is not None:
                cv2.imshow("Bird's-Eye View", bird_view)

            if frame_com_linhas is not None:
                cv2.imshow("Linhas Detectadas", frame_com_linhas)

            if edges is not None:
                cv2.imshow("Bordas Canny", edges)

            tecla = cv2.waitKey(frame_delay) & 0xFF

            if tecla == ord("q"):
                print("Processamento interrompido pelo usuário.")
                break

        video.release()
        cv2.destroyAllWindows()

        print("MatchVision encerrado com sucesso.")