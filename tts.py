import shutil
from pathlib import Path
import requests
from typing import BinaryIO

from openai import OpenAI, AsyncOpenAI
import replicate
from langdetect import detect
from TTS.api import TTS
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

from logger_config import get_logger
from config import settings

logger = get_logger(__name__)


def try_create_directory(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create directory {path}.")
        raise
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def try_open_audio_file(file_path: Path) -> BinaryIO:
    try:
        return open(file_path, "rb")
    except Exception as e:
        logger.error(f"Failed to open file at {file_path}: {e}")
        raise


class CoquiTTS:
    def __init__(self):
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tts_model = TTS(self.model_name).to(self.device)
            logger.info("Successfully loaded TTS model")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
        self.voice_path = Path(__file__).resolve().parent / "voices" / "ellie.mp3"
        self.executor = ThreadPoolExecutor()
        self.supported_languages = [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
        ]

    def run(self, text: str, file_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        try:
            lang_iso = detect(text)
            if lang_iso not in self.supported_languages:
                lang_iso = "en"
        except Exception as e:
            logger.error(f"Failed to detect language for text: {e}")
            lang_iso = "en"
        try:
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=str(self.voice_path),
                language=lang_iso,
                file_path=file_path,
            )
        except Exception as e:
            logger.error(f"Error in CoquiTTS method: {e}")
            raise

    async def arun(self, text: str, file_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        try:
            await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.run,
                text,
                file_path,
            )
        except asyncio.CancelledError:
            logger.info("Async TTS method was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Error in async TTS method: {e}")
            raise
        finally:
            logger.info("Shutting down executor.")
            self.executor.shutdown(wait=False)


class OpenAITTS:
    def __init__(self):
        self.api_key = self._get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.voice = "nova"

    def _get_openai_api_key(self):
        return settings.openai_api_key.get_secret_value()

    def run(self, text: str, file_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,  # type: ignore
                input=text,
            )
            response.write_to_file(file_path)
        except Exception as e:
            logger.error(f"Error in OpenAI TTS method: {e}")
            raise

    async def arun(self, text: str, file_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        try:
            response = await self.async_client.audio.speech.create(
                model="tts-1",
                voice=self.voice,  # type: ignore
                input=text,
            )
            response.write_to_file(file_path)
        except Exception as e:
            logger.error(f"Error in OpenAI Async TTS method: {e}")
            raise


class ReplicateTortoiseTTS:
    def __init__(self) -> None:
        self.voice_path = Path(__file__).parent / "voices" / "ellie.mp3"
        self.replicate_id = "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71"

    def run(self, text: str, file_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        custom_voice = try_open_audio_file(self.voice_path)
        input = {
            "voice_a": "custom_voice",
            "custom_voice": custom_voice,
            "text": text,
            "preset": "fast",
        }
        try:
            output_url = replicate.run(
                self.replicate_id,
                input=input,
            )
            r = requests.get(output_url)  # type: ignore
            with open(file_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            logger.error(f"Failed to run ReplicateTortoiseTTS's run method: {e}")
            raise

    async def arun(self, text: str, file_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        custom_voice = try_open_audio_file(self.voice_path)
        input = {
            "voice_a": "custom_voice",
            "custom_voice": custom_voice,
            "text": text,
            "preset": "fast",
        }
        try:
            output_url = await replicate.async_run(
                self.replicate_id,
                input=input,
            )
            r = requests.get(output_url)  # type: ignore
            with open(file_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            logger.error(f"Failed to run ReplicateTortoiseTTS's arun method: {e}")
            raise


class DummyOpenAITTS:
    def __init__(self, source_audio_file: str):
        self.voice = "nova"
        self.source_audio_file = source_audio_file

    def run(self, text: str, file_path: str):
        """Synchronously copy audio content to specified file path."""
        try_create_directory(Path(file_path).resolve().parent)
        try:
            # Copy the content of the source audio file to the specified file path
            shutil.copy(self.source_audio_file, file_path)
            logger.info("Dummy TTS file copied successfully.")
        except Exception as e:
            logger.error(f"Error in Dummy TTS method: {e}")
            raise

    async def arun(self, text: str, file_path: str):
        """Asynchronously copy audio content to specified file path."""
        try_create_directory(Path(file_path).resolve().parent)
        try:
            # Simulate async operation, if needed
            await asyncio.sleep(0)  # No delay needed, just for async syntax

            # Copy the content of the source audio file to the specified file path
            shutil.copy(self.source_audio_file, file_path)
            logger.info("Dummy Async TTS file copied successfully.")
        except Exception as e:
            logger.error(f"Error in Dummy Async TTS method: {e}")
            raise


# tts = OpenAITTS()
tts = CoquiTTS()

if __name__ == "__main__":
    source_file = "./tests/audio/moe-moe-kyun.mp3"
    tts_model = DummyOpenAITTS(source_audio_file=source_file)

    tts_model.run("Hello, this is a test.", "./tests/dummy_speech_sync.mp3")

    # For the async method, run it in an event loop
    async def main():
        await tts_model.arun("Hello, this is a test.", "./tests/dummy_speech_async.mp3")

    asyncio.run(main())
