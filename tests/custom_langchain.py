from langchain.llms.base import LLM
from langchain.chains import LLMChain
from threading import Thread
from typing import Optional
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_model_and_tokenizer(model_name="bigscience/bloom-1b7"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = initialize_model_and_tokenizer()

class CustomLangChainLLM(LLM):
    streamer: Optional[TextIteratorStreamer] = None

    def _call(self, prompt, stop=None, run_manager=None, **kwargs) -> str:
        self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
        inputs = tokenizer(prompt, return_tensors="pt")
        kwargs = dict(input_ids=inputs["input_ids"], streamer=self.streamer, max_new_tokens=20)
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"
    
