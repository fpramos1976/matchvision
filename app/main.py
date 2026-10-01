import argparse

from app.processing.video_processor import VideoProcessor

DEFAULT_VIDEO = "videos/tennis_3.mp4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Analisa um vídeo de tênis (jogadores, bola e quadra).",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"caminho do vídeo (padrão: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--classico",
        action="store_true",
        help="detecta a bola pelo método clássico (cor/movimento) em vez do TrackNet",
    )
    parser.add_argument(
        "--saida",
        help="arquivo do vídeo analisado (padrão: output/<nome>_analise.mp4)",
    )
    parser.add_argument(
        "--sem-reproducao",
        action="store_true",
        help="só processa e grava o vídeo analisado, sem reproduzir no fim",
    )
    # Mantida por compatibilidade: o TrackNet já é o padrão.
    parser.add_argument("--tracknet", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.classico:
        ball_detector = "classico"
    elif args.tracknet:
        ball_detector = "tracknet"
    else:
        ball_detector = None
    processor = VideoProcessor(
        args.video,
        ball_detector=ball_detector,
        output_path=args.saida,
        play=not args.sem_reproducao,
    )
    processor.run()


if __name__ == "__main__":
    main()
