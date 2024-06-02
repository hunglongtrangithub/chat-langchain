"""Main entrypoint for the app."""

import asyncio
import os
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from pathlib import Path
import json

from starlette.background import BackgroundTask

from logger_config import get_logger
from retriever import CustomRetriever
from custom_chat_model import CustomChatHuggingFace, DummyChat
from rag_chain import ChatRequest, RAGChain
from tts import CoquiTTS, DummyOpenAITTS
from transcription import WhisperSTT, DummyOpenAIWhisperSTT
from config import settings

logger = get_logger(__name__)


# chat_model = CustomChatHuggingFace("meta-llama/Meta-Llama-3-8B-Instruct")
# chat_model = CustomChatHuggingFace("facebook/opt-125m")
chat_model = DummyChat()
retriever = CustomRetriever(num_docs=5, semantic_ratio=0.1)
chain = RAGChain(retriever, chat_model)
# tts = CoquiTTS()
# transcribe = WhisperSTT()
tts = DummyOpenAITTS()
transcribe = DummyOpenAIWhisperSTT()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def post_processing(op, path, chunk):
    return json.dumps(
        {"ops": [{"op": op, "path": path, "value": chunk}]}, separators=(",", ":")
    )


async def astream_generator(subscription: AsyncGenerator):
    try:
        async for op, path, chunk in subscription:
            if path == "/stream-output/-":
                print(f"op: {op}, path: {path}, chunk: {chunk}")
            yield f"event: data\ndata: {post_processing(op, path, chunk)}\n\n"
            # HACK: This is a temporary fix to prevent the browser from being unable to handle the stream when it becomes too fast
            await asyncio.sleep(0.01)
    except asyncio.TimeoutError:
        error_message = "Stream timed out"
        logger.error(error_message)
        yield f"event: error\ndata: {error_message}\n\n"
    except Exception as e:
        error_message = f"Internal server error from endpoint /chat/astream_log: {e}"
        logger.error(error_message)
        yield f"event: error\ndata: {error_message}\n\n"
    finally:
        await subscription.aclose()
        yield "event: end\n\n\n"


@app.post("/chat/ainvoke_log")
async def chat(request: ChatRequest):
    try:
        response = await chain.ainvoke_log(request)
        return response
    except Exception as e:
        error_message = f"Internal server error from endpoint /chat/ainvoke_log: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )


@app.post("/chat/astream_log")
async def achat(request: ChatRequest):
    subscription = chain.astream_log(request)
    stream = astream_generator(subscription)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
    )


@app.post("/transcribe_audio")
async def transcribe_audio(
    file: UploadFile = File(...), conversationId: str = Form(...)
):
    file_name = f"{conversationId}.mp3"
    # save to local file
    upload_folder = Path(__file__).resolve().parent / "audio"
    upload_folder.mkdir(exist_ok=True)
    file_path = str(upload_folder / file_name)

    with open(file_path, "wb") as f:
        f.write(file.file.read())
    try:
        transcript = await transcribe.arun(audio_path=file_path)
        return {"transcript": transcript}
    except Exception as e:
        error_message = f"Internal server error from endpoint /transcribe_audio: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )


class MessageRequest(BaseModel):
    message: str
    conversationId: str


@app.post("/text_to_speech")
async def text_to_speech(request: MessageRequest):
    text = request.message
    speech_file_name = f"{request.conversationId}.wav"
    upload_folder = Path(__file__).resolve().parent / "audio"
    upload_folder.mkdir(exist_ok=True)
    speech_file_path = str(upload_folder / speech_file_name)

    try:
        await tts.arun(text=text, file_path=speech_file_path)
        return FileResponse(
            speech_file_path,
            background=BackgroundTask(delete_file, speech_file_path),
        )
    except Exception as e:
        error_message = f"Internal server error from endpoint /text_to_speech: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )


def delete_file(file_path: str):
    """Deletes a file from the filesystem."""
    try:
        os.remove(file_path)
        logger.info(f"Deleted file {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":
    import uvicorn

    port = settings.port
    uvicorn.run(app, host="0.0.0.0", port=port)
