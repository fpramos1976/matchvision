from app.processing.video_processor import VideoProcessor


def main():
    processor = VideoProcessor("videos/tennis_4.mp4")
    processor.run()


if __name__ == "__main__":
    main()