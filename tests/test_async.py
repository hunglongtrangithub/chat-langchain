import asyncio
import json
import time
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()


# async def stream_generator(queue: asyncio.Queue):
#     while True:
#         item = await queue.get()
#         if item is None:  # Using `None` as a signal to stop.
#             break
#         yield item


async def retrieve_docs():
    await asyncio.sleep(5)  # Simulate document retrieval delay
    formatted_docs = "<retrieved docs>"
    # result = "add", "/logs/FindDocs/final_output", str({"output": formatted_docs})
    yield "add", "/logs", {}
    yield "add", "/logs/FindDocs", {}
    yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}


async def chat_response():
    await asyncio.sleep(3)  # Simulate initial delay
    response = ""
    yield "add", "/streamed_output", []
    for chunk in range(20):
        yield "add", "/streamed_output/-", chunk
        response += str(chunk)
        await asyncio.sleep(0.1)  # Simulate delay between responses
    yield "replace", "/final_output", {"output": response}


async def stream_log():
    yield "replace", "", {}

    # Retrieve documents
    async for result in retrieve_docs():
        yield result

    # Iterate over chat_response yields
    async for result in chat_response():
        yield result


async def dummy_async_iterator(iterable):
    for item in iterable:
        await asyncio.sleep(0.1)  # Simulate an asynchronous operation
        yield f"event: data\ndata: {item}\n\n"
    yield "event: end\n"


def post_processing(op, path, chunk):
    return json.dumps(
        {"ops": [{"op": op, "path": path, "value": chunk}]}, separators=(",", ":")
    )


async def stream_generator(subscription):
    try:
        async for op, path, chunk in subscription:
            yield f"event: data\ndata: {post_processing(op, path, chunk)}\n\n"
        yield f"event: end\n\n\n"
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Stream timed out")
    finally:
        pass


@app.post("/")
async def endpoint():
    subscription = stream_log()
    return StreamingResponse(
        stream_generator(subscription), media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
