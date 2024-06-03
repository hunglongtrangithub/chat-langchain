import os
from pathlib import Path
import shutil
from logger_config import get_logger

# from sad_talker import SadTalker

# print("SadTalker imported")

logger = get_logger(__name__)


class DummyTalker:
    def __init__(self):
        logger.info("DummyTalker initialized")

    def run(self, video_path: str, audio_path: str, image_path: str):
        # copy video/chatbot1__1.mp4 to video_path
        shutil.copy(Path(__file__).parent / "video/chatbot1__1.mp4", video_path)
        logger.info(f"Created video {video_path}")
        # # combine audio_path and video_path to create video with audio
        # codec = "copy"
        # output_path = "output.mp4"
        # cmd = f"ffmpeg -i {video_path} -i {audio_path} -c:v {codec} -c:a aac {output_path}"
        # os.system(cmd)
        # logger.info(f"Created video with audio")

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


# TODO: Implement SadTalker

if __name__ == "__main__":
    talker = DummyTalker()
    talker.run(
        video_path="video/chatbot1.mp4",
        audio_path="video/chatbot1.mp3",
        image_path="faces/chatbot1.jpg",
    )
