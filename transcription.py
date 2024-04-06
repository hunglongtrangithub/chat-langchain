from typing import BinaryIO
from openai import OpenAI, AsyncOpenAI
from transformers import pipeline
import replicate
import asyncio
from pathlib import Path

from logger_config import get_logger
from concurrent.futures import ThreadPoolExecutor
from config import settings

logger = get_logger(__name__)


def try_open_audio_file(file_path: Path) -> BinaryIO:
    try:
        return open(file_path, "rb")
    except Exception as e:
        logger.error(f"Failed to open file at {file_path}: {e}")
        raise


class WhisperSTT:
    def __init__(self):
        self.transcription_model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
        )
        self.executor = ThreadPoolExecutor()

    def run(self, audio_path: str) -> str:
        try:
            transcription = self.transcription_model(audio_path)
            return transcription["text"]
        except Exception as e:
            print(f"Error in WhisperSTT method: {e}")
            raise

    async def arun(self, audio_path: str) -> str:
        try:
            transcription = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.run,
                audio_path,
            )
            return transcription
        except asyncio.CancelledError:
            logger.info("Async transcription method was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Error in async transcription method: {e}")
            raise
        finally:
            logger.info("Shutting down executor.")
            self.executor.shutdown(wait=False)


class ReplicateWhisperSTT:
    def __init__(self):
        self.replicate_id = "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"

    def run(self, audio_path: str) -> str:
        audio = try_open_audio_file(Path(audio_path))
        input = {
            "audio": audio,
            "batch_size": 64,
        }
        output = replicate.run(
            self.replicate_id,
            input=input,
        )
        return output["text"]  # type: ignore

    async def arun(self, audio_path: str) -> str:
        audio = try_open_audio_file(Path(audio_path))
        input = {
            "audio": audio,
            "batch_size": 64,
        }
        try:
            output = await replicate.async_run(
                self.replicate_id,
                input=input,
            )
            return output["text"]  # type: ignore
        except Exception as e:
            logger.error(f"Error in async Replicate WhisperSTT method: {e}")
            raise


class OpenAIWhisperSTT:
    def __init__(self):
        self.api_key = self._get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def _get_openai_api_key(self):
        return settings.openai_api_key.get_secret_value()

    def run(self, audio_path: str) -> str:
        audio = try_open_audio_file(Path(audio_path))
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
        )
        return transcription["text"]  # type: ignore

    async def arun(self, audio_path: str) -> str:
        audio = try_open_audio_file(Path(audio_path))
        try:
            transcription = await self.async_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
            )
            return transcription.text
        except Exception as e:
            logger.error(f"Error in OpenAI Async WhisperSTT method: {e}")
            raise


class DummyOpenAIWhisperSTT:
    def __init__(self, default_transcription_text="This is a dummy transcription."):
        self.default_transcription_text = (
            "This is a dummy transcription."
            if default_transcription_text is None
            else default_transcription_text
        )

    def _get_openai_api_key(self):
        # Simulate retrieval of API key
        return "dummy_api_key"

    def run(self, audio_path: str) -> str:
        # Simulate processing the audio file
        _ = try_open_audio_file(Path(audio_path))
        logger.info("Simulating synchronous transcription.")
        # Return a default dummy transcription text
        return self.default_transcription_text

    async def arun(self, audio_path: str) -> str:
        # Simulate processing the audio file asynchronously
        _ = try_open_audio_file(Path(audio_path))
        logger.info("Simulating asynchronous transcription.")
        # Simulate async delay
        await asyncio.sleep(0)  # No real delay, just for async syntax
        # Return a default dummy transcription text
        return self.default_transcription_text


# transcribe = OpenAIWhisperSTT()
transcribe = WhisperSTT()

if __name__ == "__main__":
    path_to_audio = "./tests/audio/moe-moe-kyun.mp3"
    whisper_stt = DummyOpenAIWhisperSTT()
    # Synchronous usage
    print(whisper_stt.run(path_to_audio))

    # Asynchronous usage
    async def async_main():
        transcription = await whisper_stt.arun(path_to_audio)
        print(transcription)

    asyncio.run(async_main())
