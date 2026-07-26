import cv2


def open_video(video_path: str):
    """Abre um vídeo e devolve o objeto VideoCapture."""

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Erro: não foi possível abrir o vídeo '{video_path}'.")
        return None

    print(f"Vídeo '{video_path}' aberto com sucesso.")
    return capture