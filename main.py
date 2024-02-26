"""Main entrypoint for the app."""

import asyncio
from typing import Optional, Union
from uuid import UUID

import langsmith
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langsmith import Client

from pydantic import BaseModel

from pathlib import Path
from dotenv import load_dotenv
import os

from chain import ChatRequest, answer_chain

from tts import tts
from transcription import transcribe

# TODO: implement env var checking and error handling (add schema + fail fast)
load_dotenv()

client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


add_routes(
    app, answer_chain, path="/chat", input_type=ChatRequest, config_keys=["metadata"]
)


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


class GetTraceBody(BaseModel):
    run_id: UUID


class MessageRequest(BaseModel):
    message: str
    conversationId: str


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))


@app.post("/transcribe_audio")
async def transcribe_audio(
    file: UploadFile = File(...), conversationId: str = Form(...)
):
    file_name = file.filename
    # save to local file
    upload_folder = Path(__file__).resolve().parent / "audio"
    upload_folder.mkdir(exist_ok=True)
    file_path = upload_folder / file_name
    file_path = str(file_path)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    transcript = transcribe(audio_path=file_path)

    return {"transcript": transcript, "conversationId": conversationId}


@app.post("/text_to_speech")
async def text_to_speech(
    request: MessageRequest,
    # background_tasks: BackgroundTasks,
):
    text = request.message
    speech_file_name = request.conversationId + ".mp3"
    upload_folder = Path(__file__).resolve().parent / "audio"
    upload_folder.mkdir(exist_ok=True)
    speech_file_path = upload_folder / speech_file_name

    tts(text=text, file_path=speech_file_path)

    # async def delete_file_after_delay(file_path: Path, delay: int = 30):
    #     # Wait for the delay
    #     await asyncio.sleep(delay)

    #     # Delete the file
    #     os.remove(file_path)

    # # Add a background task to delete the file after a delay
    # background_tasks.add_task(delete_file_after_delay, speech_file_path)

    return FileResponse(speech_file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
