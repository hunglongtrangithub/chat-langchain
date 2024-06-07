from pathlib import Path
import shutil
from logger_config import get_logger

from sad_talker import SadTalker

logger = get_logger(__name__)


class DummyTalker:
    def __init__(self):
        logger.info("DummyTalker initialized")

    def run(self, video_path: str, audio_path: str, image_path: str):
        # copy video/chatbot1__1.mp4 to video_path
        shutil.copy(Path(__file__).parent / "video/chatbot1__1.mp4", video_path)
        logger.info(f"Created video {video_path}")

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


class CustomSadTalker(SadTalker):
    def __init__(self):
        checkpoint_path = Path(__file__).parent / "sad_talker/checkpoints"
        config_path = Path(__file__).parent / "sad_talker/src/config"
        super().__init__(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
        )
        logger.info(f"CustomSadTalker initialized.")

    def run(
        self,
        video_path: str,
        audio_path: str,
        image_path: str,
        delete_generated_files=True,
    ):
        logger.info(f"Generating video at {video_path}")
        try:
            result_dir = str(Path(__file__).parent / "video")
            returned_video_path = super().test(
                source_image=image_path,
                driven_audio=audio_path,
                result_dir=result_dir,
            )

            # copy returned_video_path to video_path
            shutil.copy(returned_video_path, video_path)
            if delete_generated_files:
                # delete returned_video_path's parent directory
                shutil.rmtree(Path(returned_video_path).parent)
                logger.info(
                    f"Deleted generated files at {Path(returned_video_path).parent}"
                )

            logger.info(f"Created video at {video_path}")
        except Exception as e:
            error_message = f"Failed to generate video: {e}"
            logger.error(error_message)
            raise Exception(error_message)

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


if __name__ == "__main__":
    talker = CustomSadTalker()
    video_path = str(Path(__file__).parent / "video/chatbot2__1.mp4")
    audio_path = str(Path(__file__).parent / "voices/chatbot2.mp3")
    image_path = str(Path(__file__).parent / "faces/chatbot2.jpg")
    talker.run(video_path, audio_path, image_path)
