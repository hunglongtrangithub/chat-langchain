from llm_llama.model_generator.llm_pipeline import load_fine_tuned_model
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from typing import Optional, AsyncIterator, Any
from threading import Thread
import asyncio


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40


class StreamingModel:
    def __init__(
        self,
        model_name: str = None,
        model=None,
        tokenizer=None,
        stop_words: list[str] = [],
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.stop_words = stop_words
        print(f"Using device: {self.device}")

    def load(self):
        if self.model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )

    def predict(self, prompt: str, streaming=False) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_config = GenerationConfig(
            temperature=1,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            do_sample=True,
        )

        # stopping_criteria = StoppingCriteriaList(
        #     [
        #         CustomStoppingCriteria(self.tokenizer, self.stop_words),
        #     ]
        # )

        with torch.no_grad():
            generation_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "streamer": streamer if streaming else None,
                # "stopping_criteria": stopping_criteria,
            }
            if streaming:
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                def inner():
                    for text in streamer:
                        if "</s>" not in text:
                            yield text
                    thread.join()

                return inner()
            else:
                output_ids = self.model.generate(**generation_kwargs)
                output_text = self.tokenizer.batch_decode(
                    output_ids[0], skip_special_tokens=True
                )
                return output_text[0][len(prompt) + 1 :]  # Remove prompt


class RunnableLM(Runnable):
    def __init__(self, model: StreamingModel):
        super().__init__()
        self.model = model
        self.model.load()

    async def astream(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[str]:
        # Call the predict method of your model
        generator = self.model.predict(input, streaming=True)

        # Convert the generator to an async iterator and yield from it
        async for token in self._async_generator(generator):
            yield token

    async def _async_generator(self, generator):
        for value in generator:
            # Use asyncio.to_thread to run each generator iteration in a separate thread
            yield await asyncio.to_thread(lambda: value)

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> str:
        # Call the predict method of your model and return the result
        return self.model.predict(input)


model, tokenizer = load_fine_tuned_model(
    path="llm_llama/Llama-2-7b-chat_peft_128",
    peft_model=1,
)
llm = RunnableLM(StreamingModel(model=model, tokenizer=tokenizer))


if __name__ == "__main__":

    def process_output(input):
        async def inner():
            chunks = []
            async for chunk in llm.astream(input):
                chunks.append(chunk)
                print(chunk, end="|", flush=True)

        asyncio.run(inner())

    def process_output_log(input):
        async def inner():
            async for token in llm.astream_log(input):
                print(token)

        asyncio.run(inner())

    process_output("Hello, what is your name?")
    # print(chain.invoke({"topic": "cats"}))
