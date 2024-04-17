import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tts import CoquiTTS, OpenAITTS, ReplicateTortoiseTTS, BarkSuno


def test_barksuno_run():
    tts = BarkSuno()
    text = "Hello, world!"
    file_path = Path(__file__).resolve().parent / "audio" / "hello-world.mp3"
    tts.run(text, str(file_path))


def test_coqui_tts_run():
    tts = CoquiTTS()
    file_path = str(Path(__file__).parent / "audio" / "coqui_test_run.mp3")
    tts.run("Hello from Coqui TTS run method.", file_path)
    assert Path(file_path).exists()
    print(f"CoquiTTS.run test passed, saved to {file_path}")


async def test_coqui_tts_arun():
    tts = CoquiTTS()
    file_path = str(Path(__file__).parent / "audio" / "coqui_test_arun.mp3")
    await tts.arun("Hello from Coqui TTS async run method.", file_path)
    assert Path(file_path).exists()
    print(f"CoquiTTS.arun test passed, saved to {file_path}")


def test_openai_tts_run():
    tts = OpenAITTS()
    file_path = str(Path(__file__).parent / "audio" / "openai_test_run.mp3")
    tts.run("Hello from OpenAI TTS run method.", file_path)
    assert Path(file_path).exists()
    print(f"OpenAITTS.run test passed, saved to {file_path}")


async def test_openai_tts_arun():
    tts = OpenAITTS()
    file_path = str(Path(__file__).parent / "audio" / "openai_test_arun.mp3")
    await tts.arun("Hello from OpenAI TTS async run method.", file_path)
    assert Path(file_path).exists()
    print(f"OpenAITTS.arun test passed, saved to {file_path}")


def test_replicate_tortoise_tts_run():
    tts = ReplicateTortoiseTTS()
    file_path = str(Path(__file__).parent / "audio" / "replicate_test_run.mp3")
    tts.run("Hello from Replicate Tortoise TTS run method.", file_path)
    assert Path(file_path).exists()
    print(f"ReplicateTortoiseTTS.run test passed, saved to {file_path}")


async def test_replicate_tortoise_tts_arun():
    tts = ReplicateTortoiseTTS()
    file_path = str(Path(__file__).parent / "audio" / "replicate_test_arun.mp3")
    await tts.arun("Hello from Replicate Tortoise TTS async run method.", file_path)
    assert Path(file_path).exists()
    print(f"ReplicateTortoiseTTS.arun test passed, saved to {file_path}")


# Run tests
if __name__ == "__main__":
    print("Running TTS tests...")
    # print("Testing CoquiTTS...")
    # test_coqui_tts_run()
    # print("Testing CoquiTTS async...")
    # asyncio.run(test_coqui_tts_arun())
    #
    # print("Testing OpenAITTS...")
    # test_openai_tts_run()
    # print("Testing OpenAITTS async...")
    # asyncio.run(test_openai_tts_arun())
    #
    # print("Testing ReplicateTortoiseTTS...")
    # test_replicate_tortoise_tts_run()
    # print("Testing ReplicateTortoiseTTS async...")
    # asyncio.run(test_replicate_tortoise_tts_arun())
    print("Testing BarkSuno...")
    test_barksuno_run()
