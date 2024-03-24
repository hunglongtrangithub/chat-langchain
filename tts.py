from pathlib import Path
import requests
from openai import OpenAI, AsyncOpenAI
import replicate
from langdetect import detect
from TTS.api import TTS
import torch
import asyncio


class CoquiTTS:
    def __init__(self):
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model = TTS(self.model_name).to(self.device)
        self.voice_path = str(Path(__file__).parent / "voices" / "ellie.mp3")
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
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        lang_iso = detect(text)
        if lang_iso not in self.supported_languages:
            lang_iso = "en"
        self.tts_model.tts_to_file(
            text=text,
            speaker_wav=self.voice_path,
            language=lang_iso,
            file_path=file_path,
        )

    async def arun(self, text: str, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_running_loop().run_in_executor(
            None,
            self.run,
            text,
            file_path,
        )


class OpenAITTS:
    def __init__(self):
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def run(self, text: str, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        response.write_to_file(file_path)

    async def arun(self, text: str, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        response = await self.async_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        response.write_to_file(file_path)


class ReplicateTortoiseTTS:
    def __init__(self) -> None:
        self.voice_path = str(Path(__file__).parent / "voices" / "ellie.mp3")
        self.replicate_id = "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71"

    def run(self, text: str, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        custom_voice = open(self.voice_path, "rb")
        input = {
            "voice_a": "custom_voice",
            "custom_voice": custom_voice,
            "text": text,
            "preset": "fast",
        }

        output_url = replicate.run(
            self.replicate_id,
            input=input,
        )

        r = requests.get(output_url)  # type: ignore
        with open(file_path, "wb") as f:
            f.write(r.content)

    async def arun(self, text: str, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        custom_voice = open(self.voice_path, "rb")
        input = {
            "voice_a": "custom_voice",
            "custom_voice": custom_voice,
            "text": text,
            "preset": "fast",
        }
        output_url = await replicate.async_run(
            self.replicate_id,
            input=input,
        )
        r = requests.get(output_url)  # type: ignore
        with open(file_path, "wb") as f:
            f.write(r.content)


tts = OpenAITTS()


if __name__ == "__main__":
    from pathlib import Path

    text = "Hello, I am a computer program."
    file_path = str(Path(__file__).parent / "audio" / "test.mp3")
    tts.run(text, file_path)
