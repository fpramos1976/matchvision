import cv2
from video_reader import open_video
from court_detector import detect_court_lines
from manual_calibration import start_calibration


def main():
    print("Hello from MatchVision!")

    video = open_video("videos/tennis.mp4")

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
    start_calibration(frame)

    while True:
        ret, frame = video.read()

        if not ret:
            print("Fim do vídeo ou erro ao ler o frame.")
            break

        # O detector agora devolve o frame com as linhas desenhadas e o Canny
        frame_com_linhas, edges = detect_court_lines(frame)

        # Exibe o resultado do Canny
        if edges is not None:
            cv2.imshow("MatchVision - Bordas Canny", edges)

        # Exibe o frame com as linhas detectadas
        if frame_com_linhas is not None:
            cv2.imshow("MatchVision - Linhas Detectadas", frame_com_linhas)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Processamento interrompido pelo usuário.")
            break

    video.release()
    cv2.destroyAllWindows()
    print("MatchVision encerrado com sucesso.")


if __name__ == "__main__":
    main()