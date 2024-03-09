"""Main entrypoint for the app."""

import asyncio
from typing import Optional, Union
from uuid import UUID

import langsmith
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langsmith import Client
from openai import OpenAI
from pydantic import BaseModel
import json

from pathlib import Path
from dotenv import load_dotenv
import os

from chain import ChatRequest, answer_chain

load_dotenv()

client = Client()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


# @app.post("/chat/stream_log")
# async def chat(request: ChatRequest):
#     async def log_messages():
#         async for message in answer_chain.astream_log(request):
#             yield f"data: {json.dumps(message)}\n\n"

#     return StreamingResponse(log_messages(), media_type="text/event-stream")


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
    )
    return {"result": "patched feedback successfully", "code": 200}


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
    upload_folder = Path(__file__).resolve().parent / "audio"
    upload_folder.mkdir(exist_ok=True)
    file_path = upload_folder / file.filename

    with open(file_path, "wb") as f:
        f.write(file.file.read())
    file_data = open(file_path, "rb")

    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=file_data,
        response_format="text",
    )

    file_data.close()
    file_path.unlink()

    return {"transcript": transcript, "conversationId": conversationId}


class MessageRequest(BaseModel):
    message: str
    conversationId: str


@app.post("/text_to_speech")
async def text_to_speech(request: MessageRequest):
    text = request.message
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    response.write_to_file(speech_file_path)

    return FileResponse(speech_file_path, filename="speech.mp3")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
