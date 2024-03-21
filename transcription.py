from pathlib import Path
from transformers import pipeline
import librosa

transcription_model = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3"
)


def transcribe(audio_path: str) -> str:
    transcription = transcription_model(audio_path)["text"]
    return transcription


if __name__ == "__main__":
    # audio_path = "audio/test.mp3"
    # audio_path = "audio/output.wav"
    audio_path = "voices/MyShell_chat_24-01-21_15_37_50_Hutao.mp3"
    transcript = transcribe(audio_path)
    print(transcript)
