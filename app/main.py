import cv2
from video_reader import open_video


def main():
    print("Hello from MatchVision!")

    # 1. Abre o arquivo de vídeo
    video = open_video("videos/tennis_1.mp4")

    if video is None:
        print("Encerrando o MatchVision.")
        return

    print("Pronto para começar a processar os frames!")

    # 2. Loop para ler e exibir o vídeo frame por frame
    while True:
        # read() devolve um booleano (ret) e a imagem do frame atual (frame)
        ret, frame = video.read()

        # Se ret for False, significa que o vídeo acabou ou houve um erro
        if not ret:
            print("Fim do vídeo ou erro ao ler o frame.")
            break

        # Exibe o frame em uma janela chamada "MatchVision - Tennis"
        cv2.imshow("MatchVision - Tennis", frame)

        # Aguarda 30 milissegundos entre os frames. 
        # Se o usuário pressionar a tecla 'q', o loop fecha.
        # Errado
        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Processamento interrompido pelo usuário.")
            break

    # 3. Limpa a memória e fecha as janelas ao terminar
    video.release()
    cv2.destroyAllWindows()
    print("MatchVision encerrado com sucesso.")


if __name__ == "__main__":
    main()