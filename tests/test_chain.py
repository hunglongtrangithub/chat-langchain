import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import asyncio
from rag_chain import ChatRequest, chain, chain_without_memory
import time
from contextlib import contextmanager

@contextmanager
def timeit():
    start = time.time()
    yield
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

current_chat = {
    "question": "Can you give me examples of applications built with LangChain?",
    "chat_history": [
        {
            "human": "What is LangChain?",
            "ai": "LangChain is a framework designed to facilitate the building of applications that leverage large language models (LLMs) like GPT. It provides tools and abstractions to make it easier to create complex systems involving language understanding and generation.",
        },
        {
            "human": "How does LangChain work?",
            "ai": "LangChain works by providing a set of libraries and APIs that allow developers to easily integrate language models into their applications. It supports various components for dialogue management, document retrieval, and information synthesis, making it versatile for a range of use cases.",
        },
        {
            "human": "What are some key features of LangChain?",
            "ai": "Key features of LangChain include modular components for building chat systems, document retrieval and summarization tools, and integration capabilities with multiple language models and data sources. It's designed to be flexible and extensible, allowing developers to customize their applications as needed.",
        },
        {
            "human": "Can LangChain be used for building chatbots?",
            "ai": "Yes, LangChain can be used for building chatbots. It offers specialized components for handling conversations, managing context, and generating responses, making it a powerful tool for developers looking to create advanced chatbot applications.",
        },
        {
            "human": "How can LangChain integrate with other technologies?",
            "ai": "LangChain can integrate with other technologies through its flexible architecture. It supports connecting with various APIs, data sources, and language models, allowing for seamless integration with existing systems and tools. This makes it suitable for a wide range of applications beyond just language processing.",
        },
    ],
}

request = ChatRequest(**current_chat)


def test_chain():
    for chunk in chain.stream_log(request):
        print(chunk, end="\n", flush=True)


def test_chain_async():
    async def print_stream(request):
        async for chunk in chain.astream_log(request):
            print(chunk, end="\n", flush=True)

    asyncio.run(print_stream(request))


def test_chain_without_memory():
    for chunk in chain_without_memory.stream_log(request):
        print(chunk, end="\n", flush=True)


def test_chain_without_memory_async():
    async def print_stream(request):
        async for chunk in chain_without_memory.astream_log(request):
            print(chunk, end="\n", flush=True)

    asyncio.run(print_stream(request))


if __name__ == "__main__":
    # with timeit():
    #     test_chain()
    # with timeit():
    #     test_chain_async()
    with timeit():
        test_chain_without_memory()
    with timeit():
        test_chain_without_memory_async()
    print("Done")
