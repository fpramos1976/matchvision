import cv2

from app import configuration
from app.utils.video_reader import open_video
from app.calibration.court_detector import detect_court_lines
from app.calibration.manual_calibration import ManualCalibration
from app.utils.configuration import get_match_configuration

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def run(self):
        print("Hello from MatchVision!")

        configuration = get_match_configuration()

        print(f"Tipo da partida: {configuration.court_type.value}")

        match_config = get_match_configuration()

        video = open_video(self.video_path)

        if video is None:
            print("Encerrando o MatchVision.")
            return

        print("Pronto para começar a processar os frames!")

        # Lê o primeiro frame
        ret, frame = video.read()

        if not ret:
            print("Erro ao ler o primeiro frame.")
            video.release()
            return

        # Inicia a calibração manual

        calibration = ManualCalibration(frame, configuration)
        points = calibration.run()
        print(f"Pontos de calibração selecionados: {points}")


        while True:
            ret, frame = video.read()

            if not ret:
                print("Fim do vídeo ou erro ao ler o frame.")
                break

            frame_com_linhas, edges = detect_court_lines(frame)

            if edges is not None:
                cv2.imshow("MatchVision - Bordas Canny", edges)

            if frame_com_linhas is not None:
                cv2.imshow("MatchVision - Linhas Detectadas", frame_com_linhas)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("Processamento interrompido pelo usuário.")
                break

        video.release()
        cv2.destroyAllWindows()
        print("MatchVision encerrado com sucesso.")