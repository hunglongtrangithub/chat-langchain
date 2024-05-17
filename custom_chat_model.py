from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from typing import Dict, AsyncGenerator, Generator, Optional, Union, List
from jinja2.exceptions import TemplateError

from huggingface_hub import login
from threading import Thread
import torch
import openai
import replicate
import groq

from logger_config import get_logger
from config import settings
from chat_templates import CHAT_TEMPLATES

logger = get_logger(__name__)


class BaseChat:
    def __init__(self):
        logger.info("BaseChat initialized.")

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        return "This is a dummy response."

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        yield "This is a dummy response."

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        return "This is a dummy response."

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        yield "This is a dummy response."


class CustomChatHuggingFace(BaseChat):
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
        self._huggingface_login()
        self.device = self._determine_device()
        self.checkpoint = checkpoint
        try:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(self.checkpoint)
                if not tokenizer
                else tokenizer
            )
            self.tokenizer.chat_template = CHAT_TEMPLATES.get(
                self.checkpoint, self.tokenizer.default_chat_template
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        try:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                if not model
                else model
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.max_length = 128  # can modify this to be the max length of the model

        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs
        if self.checkpoint == "meta-llama/Meta-Llama-3-8B-Instruct":
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            self.generation_kwargs["eos_token_id"] = terminators  # type: ignore

        logger.info(f"{checkpoint} initialized.")
        logger.info("Initialized on device: {}".format(self.device))
        if self.device == "cuda":
            logger.info("Number of GPUs: {}".format(torch.cuda.device_count()))
            logger.info(
                "Memory footprint: {:.2f}GB".format(
                    self.model.get_memory_footprint() / 1024**3
                )
            )

    def _huggingface_login(self):
        token = settings.hf_token.get_secret_value()
        login(token=token)

    def _determine_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            chat_history = self.tokenizer.apply_chat_template(
                current_conversation,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except TemplateError as e:
            logger.error(f"Error applying chat template: {e}")
            raise

        tokenized_chat_history = self.tokenizer(
            text=chat_history, return_tensors="pt"  # type: ignore
        ).to(self.device)

        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][  # type: ignore
            :, -self.max_length :
        ]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history[  # type: ignore
            "attention_mask"
        ][
            :, -self.max_length :
        ]

        prompt_size = tokenized_chat_history["input_ids"].shape[1]  # type: ignore
        try:
            generated_tokens = self.model.generate(
                **tokenized_chat_history,
                **self.generation_kwargs,
            )
            # skip the prompt tokens and decode the rest
            response = self.tokenizer.decode(
                generated_tokens[:, prompt_size:][0], skip_special_tokens=True
            )
            return response
        except Exception as e:
            logger.error(
                f"Error generating tokens: {e}",
                exc_info=True,
                stack_info=True,
            )
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
        )
        chat_history = self.tokenizer.apply_chat_template(
            current_conversation,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        tokenized_chat_history = self.tokenizer(
            text=chat_history, return_tensors="pt"  # type: ignore
        ).to(self.device)

        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][  # type: ignore
            :, -self.max_length :
        ]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history[  # type: ignore
            "attention_mask"
        ][
            :, -self.max_length :
        ]

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
            try:
                for new_text in streamer:
                    yield new_text
            except Exception as e:
                logger.error(
                    f"Error in stream generator: {e}",
                    exc_info=True,
                    stack_info=True,
                )
                raise
            finally:
                thread.join()

        return generator()

    async def ainvoke(self, current_conversation: list[dict[str, str]]) -> str:
        try:
            return self.invoke(current_conversation)
        except Exception as e:
            logger.error(
                f"Error in ainvoke generator: {e}",
            )
            raise

    async def astream(
        self, current_conversation: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        try:
            for item in self.stream(current_conversation):
                yield item
        except Exception as e:
            logger.error(
                f"Error in astream generator: {e}",
            )
            raise


class CustomChatOpenAI(BaseChat):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.api_key = self._get_openai_api_key()
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model_name = model_name
        logger.info("CustomChatOpenAI initialized.")

    def _get_openai_api_key(self):
        openai_api_key = settings.openai_api_key.get_secret_value()
        return openai_api_key

    def _handle_api_error(self, e: Exception):
        """Centralized error handling for OpenAI API errors."""
        if isinstance(e, openai.APITimeoutError):
            logger.error("The request took too long: %s", e)
        elif isinstance(e, openai.APIConnectionError):
            logger.error(
                "Request could not reach the OpenAI API servers or establish a secure connection: %s",
                e,
            )
        elif isinstance(e, openai.AuthenticationError):
            logger.error("API key or token is invalid, expired, or revoked: %s", e)
        elif isinstance(e, openai.InternalServerError):
            logger.error("OpenAI API internal server error: %s", e)
        elif isinstance(e, openai.RateLimitError):
            logger.error("Rate limit reached: %s", e)
        else:
            logger.error("An unexpected error occurred: %s", e)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                stream=True,
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                stream=True,
            )
            async for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            self._handle_api_error(e)
            raise


class CustomChatLlamaReplicate(BaseChat):
    default_system_prompt = "You are a helpful assistant."

    def __init__(self, model_name: str = "meta/llama-2-70b-chat"):
        self.model_name = model_name
        logger.info("CustomChatLlamaReplicate initilized.")

    def _handle_error(self, error: Exception, context: str = ""):
        if context:
            logger.error(f"Error during {context}: {error}")
        else:
            logger.error(f"Error: {error}")

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
        try:
            output = replicate.run(
                self.model_name,
                input={
                    "prompt": self.process_chat(current_conversation),
                    "max_new_tokens": 512,
                    "temperature": 0.75,
                },
            )
            return "".join(output)
        except Exception as e:
            self._handle_error(e, context="invoke")
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        try:
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
                elif chunk.event.value == "error":
                    logger.error(f"Error in Replicate stream: {chunk.data}")
                elif chunk.event.value == "done":
                    logger.info(f"Replicate stream done: {chunk.data or 'successful'}")
        except Exception as e:
            self._handle_error(e, context="stream")
            raise

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
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
                elif chunk.event.value == "error":
                    logger.error(f"Error in Replicate stream: {chunk.data}")
                elif chunk.event.value == "done":
                    logger.info(f"Replicate stream done: {chunk.data or 'successful'}")
            return response
        except Exception as e:
            self._handle_error(e, context="ainvoke")
            raise

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        try:
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
                elif chunk.event.value == "error":
                    logger.error(f"Error in Replicate stream: {chunk.data}")
                elif chunk.event.value == "done":
                    logger.info(f"Replicate stream done: {chunk.data or 'successful'}")
        except Exception as e:
            self._handle_error(e, context="astream")
            raise


class CustomChatGroq(BaseChat):
    default_generation_kwargs = {
        "temperature": 0.5,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stop": [],
    }

    def __init__(
        self,
        model_name: str = "gemma-7b-it",
        generation_kwargs: Optional[dict] = None,
    ):
        self.api_key = self._get_groq_api_key()
        self.client = groq.Groq(api_key=self.api_key)
        self.async_client = groq.AsyncGroq(api_key=self.api_key)
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs
        logger.info("CustomChatGroq initialized.")

    def _get_groq_api_key(self):
        groq_api_key = settings.groq_api_key.get_secret_value()
        return groq_api_key

    def _handle_api_error(self, e: Exception):
        """Centralized error handling for Groq API errors."""
        if isinstance(e, groq.APITimeoutError):
            logger.error("The request took too long: %s", e)
        elif isinstance(e, groq.APIConnectionError):
            logger.error(
                "Request could not reach the Groq API servers or establish a secure connection: %s",
                e,
            )
        elif isinstance(e, groq.AuthenticationError):
            logger.error("API key or token is invalid, expired, or revoked: %s", e)
        elif isinstance(e, groq.InternalServerError):
            logger.error("Groq API internal server error: %s", e)
        elif isinstance(e, groq.RateLimitError):
            logger.error("Rate limit reached: %s", e)
        else:
            logger.error("An unexpected error occurred: %s", e)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
                stream=True,
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
                stream=True,
            )
            async for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            self._handle_api_error(e)
            raise


# CHECKPOINT = "facebook/opt-125m"
# CHECKPOINT = "meta-llama/Llama-2-7b-chat-hf"
# CHECKPOINT = "georgesung/llama2_7b_chat_uncensored"
# CHECKPOINT = "Tap-M/Luna-AI-Llama2-Uncensored"

# from llm_llama.model_generator.llm_pipeline import load_fine_tuned_model
# from pathlib import Path
#
# CHECKPOINT = Path(__file__).parent.resolve() / "llm_llama/Llama-2-7b-chat_peft_128"
# model, tokenizer = load_fine_tuned_model(CHECKPOINT, peft_model=1)
# chat_llm = CustomChatHuggingFace(model=model, tokenizer=tokenizer)

# chat_llm = CustomChatLlamaReplicate()

# chat_llm = CustomChatOpenAI()
# chat_llm = CustomChatHuggingFace()

# model_name = "mixtral-8x7b-32768"
# model_name = "llama3-8b-8192"
# chat_llm = CustomChatGroq(model_name=model_name)
