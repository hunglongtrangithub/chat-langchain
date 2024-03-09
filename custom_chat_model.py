from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
from threading import Thread


class CustomModel:
    default_generation_kwargs = {
        "max_new_tokens": 1024,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.1,
    }

    def __init__(self, checkpoint: str, generation_kwargs: dict = None):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.pipe = pipeline("text-generation", checkpoint)
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs

    def invoke(self, input_text: str):
        return self.pipe(input_text, **self.generation_kwargs)[0]["generated_text"]

    def stream(self, input_text: str):
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")
        print(inputs)
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


class CustomChatModel:
    default_generation_kwargs = {
        "max_new_tokens": 1024,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    }

    def __init__(
        self,
        checkpoint: str,
        system_prompt: str = None,
        generation_kwargs: dict = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.pipe = pipeline("text-generation", checkpoint)

        self.chat_history = []
        if system_prompt is not None:
            self.chat_history.append({"role": "system", "content": system_prompt})
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs

    def invoke(self, input_text: str):
        self.chat_history.append({"role": "user", "content": input_text})
        response = self.pipe(self.chat_history, **self.generation_kwargs)[0][
            "generated_text"
        ][-1]["content"]
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def stream(self, input_text: str):
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.chat_history.append({"role": "user", "content": input_text})
        tokenized_chat_history = self.tokenizer.apply_chat_template(
            self.chat_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
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
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                yield new_text
            self.chat_history.append({"role": "assistant", "content": full_response})
            thread.join()

        return generator()
