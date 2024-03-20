from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
import torch
from openai import OpenAI
from threading import Thread
from typing import Dict, Generator, Optional, Union, List
from pydantic import BaseModel, root_validator


# This class is for reference only, it is not used in the final implementation
class CustomModel:
    default_generation_kwargs = {
        "max_new_tokens": 512,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.1,
    }

    def __init__(
        self,
        checkpoint: str,
        generation_kwargs: Optional[Dict[str, Union[int, bool, float]]] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs
        self.max_length = 512  # can modify this to be the max length of the model

        print("CustomChatModel initialized.")
        print(
            "Device:",
            (
                str(torch.cuda.device_count()) + " gpus"
                if torch.cuda.is_available()
                else "cpu"
            ),
        )
        print("Checkpoint:", checkpoint)

    def invoke(self, input_text: str) -> str:
        input_tokens = self.tokenizer(text=input_text, return_tensors="pt").to(self.device)

        # truncate inputs to max_length of model (take the last self.max_length tokens)
        input_tokens["input_ids"] = input_tokens["input_ids"][:, -self.max_length :]
        input_tokens["attention_mask"] = input_tokens["attention_mask"][:, -self.max_length :]

        # get the size of the prompt
        prompt_size = input_tokens["input_ids"].shape[1]
        print("Prompt size:", prompt_size)

        generated_tokens = self.model.generate(
            **input_tokens,
            **self.generation_kwargs,
        )

        # skip the prompt tokens and decode the rest
        response = self.tokenizer.decode(
            generated_tokens[:, prompt_size:][0], skip_special_tokens=True
        )
        return response

    def stream(self, input_text: str) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        input_tokens = self.tokenizer(text=input_text, return_tensors="pt").to(self.device)

        # truncate inputs to max_length of model (take the last self.max_length tokens)
        input_tokens["input_ids"] = input_tokens["input_ids"][:, -self.max_length :]
        input_tokens["attention_mask"] = input_tokens["attention_mask"][:, -self.max_length :]

        # get the size of the prompt
        prompt_size = input_tokens["input_ids"].shape[1]
        print("Prompt size:", prompt_size)

        thread = Thread(
            target=self.model.generate,
            kwargs={
                **input_tokens,
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


class Message(BaseModel):
    role: str
    content: str

    @root_validator(pre=True)
    def validate_role(cls, values):
        role = values.get("role")
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be either 'system', user' or 'assistant'")
        return values


class CustomChatModel:
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
        checkpoint: str,
        generation_kwargs: Optional[Dict[str, Union[int, bool, float]]] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.max_length = 512  # can modify this to be the max length of the model
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs

        print("CustomChatModel initialized.")
        print(
            "Device:",
            (
                str(torch.cuda.device_count()) + " gpus"
                if torch.cuda.is_available()
                else "cpu"
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
        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][:, -self.max_length :]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history["attention_mask"][:, -self.max_length :]

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
        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][:, -self.max_length :]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history["attention_mask"][:, -self.max_length :]

        # get the size of the prompt
        prompt_size = tokenized_chat_history["input_ids"].shape[1]
        # print("Prompt size:", prompt_size)

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

    def __call__(
        self, message_list: List[Message], stream=False
    ) -> Union[str, Generator[str, None, None]]:
        # unpack the message_list into a list of dictionaries
        current_conversation = [
            {"role": message.role, "content": message.content}
            for message in message_list
        ]
        if current_conversation[-1]["role"] != "user":
            raise ValueError(
                "The last message in the conversation must be from the user"
            )
        if stream:
            return self.stream(current_conversation)
        else:
            return self.invoke(current_conversation)


class CustomChatOpenAI:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model_name = model_name

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=current_conversation,  # type: ignore
        )
        return completion.choices[0].message.content  # type: ignore

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=current_conversation,  # type: ignore
            stream=True,
        )
        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def __call__(
        self, message_list: List[Message], stream=False
    ) -> Union[str, Generator[str, None, None]]:
        # unpack the message_list into a list of dictionaries
        current_conversation = [
            {"role": message.role, "content": message.content}
            for message in message_list
        ]
        if current_conversation[-1]["role"] != "user":
            raise ValueError(
                "The last message in the conversation must be from the user"
            )
        if stream:
            return self.stream(current_conversation)
        else:
            return self.invoke(current_conversation)


# This class is for reference only, it is not used in the final implementation
class CustomOpenAI:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model_name = model_name

    def invoke(self, input_text: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
        )
        return completion.choices[0].message.content  # type: ignore

    def stream(self, input_text: str) -> Generator[str, None, None]:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            stream=True,
        )
        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                yield token


# if __name__ == "__main__":
#     # model = CustomModel("facebook/opt-125m")
#     # print(model.invoke("Hello, how are you?"))
#     openai_model = CustomOpenAI()
#     # print(openai_model.invoke("Hello, how are you?"))
#     for chunk in openai_model.stream("Hello, how are you?"):
#         print(chunk, end="", flush=True)
