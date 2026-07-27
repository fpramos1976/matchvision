import cv2

from app.utils.video_reader import open_video
from app.utils.configuration import get_match_configuration

from app.calibration.manual_calibration import ManualCalibration
from app.calibration.homography import Homography

# Temporário (iremos remover quando a Bird's-Eye View estiver pronta)
from app.calibration.court_detector import detect_court_lines


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
        homography = Homography(points, configuration)

        homography.compute()

        print("Homografia calculada!")

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

        # ==========================================
        # Processamento
        # ==========================================
        contador = 0

        while True:

            contador += 1

            ret, frame = video.read()

            if not ret:
                print("Fim do vídeo.")
                break

            # Bird's-Eye View (por enquanto devolve o frame original)
            bird_view = homography.transform(frame)

            # Detector antigo (apenas para comparação)
            frame_com_linhas, edges = detect_court_lines(frame)

            # Janelas
            cv2.imshow("Original", frame)

            if bird_view is not None:
                cv2.imshow("Bird's-Eye View", bird_view)

            if frame_com_linhas is not None:
                cv2.imshow("Linhas Detectadas", frame_com_linhas)

            if edges is not None:
                cv2.imshow("Bordas Canny", edges)

            tecla = cv2.waitKey(30) & 0xFF

            if tecla == ord("q"):
                print("Processamento interrompido pelo usuário.")
                break

        video.release()
        cv2.destroyAllWindows()

        print("MatchVision encerrado com sucesso.")