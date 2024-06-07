import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from talking_face import CustomSadTalker

talker = CustomSadTalker()


def test_talker():
    video_path = str(Path(__file__).parent.parent / "video/chatbot1__1.mp4")
    audio_path = str(Path(__file__).parent.parent / "voices/chatbot1.mp3")
    image_path = str(Path(__file__).parent.parent / "faces/chatbot1.jpg")
    print(video_path, audio_path, image_path)
    talker.run(video_path, audio_path, image_path)
    assert os.path.exists(video_path)
    print("Test passed")


if __name__ == "__main__":
    test_talker()
