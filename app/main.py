from video_reader import open_video


def main():
    print("Hello from MatchVision!")

    video = open_video("videos/tennis.mp4")

    if video is None:
        print("Encerrando o MatchVision.")
        return

    print("Pronto para começar a processar os frames!")


if __name__ == "__main__":
    main()