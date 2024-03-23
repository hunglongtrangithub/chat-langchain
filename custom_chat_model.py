from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
import torch
from openai import OpenAI, AsyncOpenAI
import replicate
from threading import Thread
from typing import Dict, AsyncGenerator, Generator, Optional, Union, List
import asyncio
from concurrent.futures import ThreadPoolExecutor


class CustomChatHuggingFace:
    default_generation_kwargs = {
        # "truncation": True,
        # "max_length": 512,
        "max_new_tokens": 512,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    }

    def __init__(
        self,
        checkpoint: str = "meta-llama/Llama-2-7b-chat-hf",
        model=None,
        tokenizer=None,
        generation_kwargs: Optional[Dict[str, Union[int, bool, float]]] = None,
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer = (
            AutoTokenizer.from_pretrained(checkpoint) if not tokenizer else tokenizer
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
            if not model
            else model
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.max_length = 512  # can modify this to be the max length of the model
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs
        # Executor for running synchronous methods asynchronously
        self.executor = ThreadPoolExecutor()

        print("CustomChatModel initialized.")
        print(
            "Device:",
            (
                str(torch.cuda.device_count()) + " gpus"
                if torch.cuda.is_available()
                else self.device
            ),
        )
        print("Checkpoint:", checkpoint)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        chat_history = self.tokenizer.apply_chat_template(
            current_conversation,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        tokenized_chat_history = self.tokenizer(
            text=chat_history, return_tensors="pt"
        ).to(self.device)

        # truncate inputs to max_length of model (take the last self.max_length tokens)
        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][
            :, -self.max_length :
        ]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history[
            "attention_mask"
        ][:, -self.max_length :]

        # get the size of the prompt
        prompt_size = tokenized_chat_history["input_ids"].shape[1]
        # print("Prompt size:", prompt_size)

        generated_tokens = self.model.generate(
            **tokenized_chat_history,
            **self.generation_kwargs,
        )

        # skip the prompt tokens and decode the rest
        response = self.tokenizer.decode(
            generated_tokens[:, prompt_size:][0], skip_special_tokens=True
        )
        return response

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        chat_history = self.tokenizer.apply_chat_template(
            current_conversation,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        tokenized_chat_history = self.tokenizer(
            text=chat_history, return_tensors="pt"
        ).to(self.device)

        # truncate inputs to max_length of model (take the last self.max_length tokens)
        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][
            :, -self.max_length :
        ]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history[
            "attention_mask"
        ][:, -self.max_length :]

        thread = Thread(
            target=self.model.generate,
            kwargs={
                **tokenized_chat_history,
                "streamer": streamer,
                **self.generation_kwargs,
            },
        )
        thread.start()

        def generator():
            for new_text in streamer:
                yield new_text
            thread.join()

        return generator()

    async def ainvoke(self, current_conversation: list[dict[str, str]]) -> str:
        loop = asyncio.get_running_loop()
        # offload the synchronous invoke method to the executor to run it in a separate thread
        response = await loop.run_in_executor(
            self.executor, self.invoke, current_conversation
        )
        return response

    async def astream(
        self, current_conversation: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()

        # A queue to communicate between the background task and the async generator
        queue = asyncio.Queue()

        def background_task():
            for item in self.stream(current_conversation):
                loop.call_soon_threadsafe(queue.put_nowait, item)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # Signal completion

        # Start the synchronous generator in a separate thread
        loop.run_in_executor(self.executor, background_task)

        # Async generator to yield items back in the async context
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item


class CustomChatOpenAI:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.model_name = model_name
        print("CustomChatOpenAI initialized.")

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=current_conversation,  # type: ignore
        )
        return completion.choices[0].message.content or ""

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=current_conversation,  # type: ignore
            stream=True,
        )
        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            yield token

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=current_conversation,  # type: ignore
        )
        return completion.choices[0].message.content or ""

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=current_conversation,  # type: ignore
            stream=True,
        )
        async for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            yield token


class CustomChatLlamaReplicate:
    default_system_prompt = "You are a helpful assistant."

    def __init__(self, model_name: str = "meta/llama-2-70b-chat"):
        self.model_name = model_name
        print("CustomChatLlamaReplicate initilized.")

    def process_chat(self, chat: List[Dict[str, str]]):
        system_prompt = (
            chat[0]["content"]
            if chat[0]["role"] == "system"
            else self.default_system_prompt
        )
        parts = []

        for turn in chat:
            if turn["role"] == "system":
                system_prompt = turn["content"]
                continue

            if turn["role"] == "user":
                if system_prompt != "":
                    parts.append(
                        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{turn['content']} [/INST]"
                    )
                    system_prompt = ""
                else:
                    parts.append(f"<s>[INST] {turn['content']} [/INST]")

            if turn["role"] == "assistant":
                parts.append(f" {turn['content']} </s>")

        return "".join(parts)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        output = replicate.run(
            self.model_name,
            input={
                "prompt": self.process_chat(current_conversation),
                "max_new_tokens": 512,
                "temperature": 0.75,
            },
        )
        return "".join(output)

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        output = replicate.stream(
            self.model_name,
            input={
                "prompt": self.process_chat(current_conversation),
                "max_new_tokens": 512,
                "temperature": 0.75,
            },
        )
        for chunk in output:
            if chunk.event.value == "output":
                yield chunk.data

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        output = await replicate.async_run(
            self.model_name,
            input={
                "prompt": self.process_chat(current_conversation),
                "max_new_tokens": 512,
                "temperature": 0.75,
            },
        )
        response = ""
        async for chunk in output:
            if chunk.event.value == "output":
                response += chunk.data
        return response

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        output = await replicate.async_stream(
            self.model_name,
            input={
                "prompt": self.process_chat(current_conversation),
                "max_new_tokens": 512,
                "temperature": 0.75,
            },
        )
        async for chunk in output:
            if chunk.event.value == "output":
                yield chunk.data
