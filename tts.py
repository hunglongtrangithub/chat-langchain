import requests
from dotenv import load_dotenv
import os

from langdetect import detect
from TTS.api import TTS
import torch

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def tts(text, file_path):
    supported_languages = [
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
    lang_iso = detect(text)
    if lang_iso not in supported_languages:
        lang_iso = "en"
    tts_model.tts_to_file(
        text=text,
        speaker_wav="voices/ellie.mp3",
        language=lang_iso,
        file_path=file_path,
    )


if __name__ == "__main__":
    from pathlib import Path

    text = "Hello, I am a computer program."
    file_path = Path(__file__).parent / "audio" / "test.mp3"
    tts(text, file_path)
    print(f"Saved TTS to {file_path}")
