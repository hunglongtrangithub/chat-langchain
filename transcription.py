from openai import OpenAI, AsyncOpenAI
from transformers import pipeline
import replicate
import asyncio


class WhisperSTT:
    def __init__(self):
        self.transcription_model = pipeline(
            "automatic-speech-recognition", model="openai/whisper-large-v3"
        )

    def run(self, audio_path: str) -> str:
        transcription = self.transcription_model(audio_path)
        return transcription  # type: ignore

    async def arun(self, audio_path: str) -> str:
        transcription = await asyncio.get_event_loop().run_in_executor(
            None,
            self.run,
            audio_path,
        )
        return transcription


class ReplicateWhisperSTT:
    def __init__(self):
        self.replicate_id = "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"

    def run(self, audio_path: str) -> str:
        audio = open(audio_path, "rb")
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
        audio = open(audio_path, "rb")
        input = {
            "audio": audio,
            "batch_size": 64,
        }
        output = await replicate.async_run(
            self.replicate_id,
            input=input,
        )
        return output["text"]  # type: ignore


class OpenAIWhisperSTT:
    def __init__(self):
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def run(self, audio_path: str) -> str:
        audio = open(audio_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
        )
        return transcription["text"]  # type: ignore

    async def arun(self, audio_path: str) -> str:
        audio = open(audio_path, "rb")
        transcription = await self.async_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
        )
        return transcription.text


transcribe = OpenAIWhisperSTT()


if __name__ == "__main__":
    audio_path = "voices/MyShell_chat_24-01-21_15_37_50_Hutao.mp3"
    transcript = transcribe.run(audio_path)
    print(transcript)
