from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
from openai import OpenAI
from threading import Thread
from typing import Dict, Generator, Optional, Union, List
from pydantic import BaseModel, root_validator


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
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.pipe = pipeline("text-generation", checkpoint)
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs

    def invoke(self, input_text: str) -> str:
        result = self.pipe(
            input_text,
            **self.generation_kwargs,
            # truncation=True,
            # max_length=512,
        )  # type: ignore
        response = result[0]["generated_text"]  # type: ignore
        return response  # type: ignore

    def stream(self, input_text: str) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            # truncation=True,
            # max_length=512,
        )
        # truncate inputs to max_length of model
        # print(inputs)
        thread = Thread(
            target=self.model.generate,
            kwargs={
                **inputs,
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
    default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

    def __init__(
        self,
        checkpoint: str,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Union[int, bool, float]]] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.pipe = pipeline("text-generation", checkpoint)

        self.system_prompt = system_prompt or self.default_system_prompt
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        chat_history = [
            {"role": "system", "content": self.system_prompt}
        ] + current_conversation
        result = self.pipe(
            chat_history,
            **self.generation_kwargs,
            # truncation=True,
            # max_length=512,
        )  # type: ignore
        response = result[0]["generated_text"][-1]["content"]  # type: ignore
        return response  # type: ignore

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        chat_history = [
            {"role": "system", "content": self.system_prompt}
        ] + current_conversation
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        tokenized_chat_history = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            # truncation=True,
            # max_length=512,
        )

        thread = Thread(
            target=self.model.generate,
            kwargs={
                "inputs": tokenized_chat_history,
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
