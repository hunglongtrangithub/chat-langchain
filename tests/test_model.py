import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).resolve().parent.parent))
from custom_chat_model import (
    CustomChatHuggingFace,
    CustomChatLlamaReplicate,
    CustomChatOpenAI,
    CustomChatGroq,
)

checkpoint = "facebook/opt-125m"
# checkpoint = "meta-llama/Llama-2-7b-chat-hf"


def test_hf_model():
    model = CustomChatHuggingFace(checkpoint)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_llama_model(stream=False):
    model = CustomChatLlamaReplicate(checkpoint)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_openai_model(stream=False):
    model = CustomChatOpenAI()
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_groq_model(stream=False):
    model = CustomChatGroq()
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


if __name__ == "__main__":
    # print("Testing Hugging Face model")
    # test_hf_model()
    # print("Testing Llama model")
    # test_llama_model()
    # print("Testing OpenAI model")
    # test_openai_model()
    print("Testing Groq model")
    test_groq_model()
    print("All tests passed")
