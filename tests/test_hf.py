import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# checkpoint = "facebook/opt-125m"
checkpoint = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def test_chat_template():
    messages = [
        {
            "role": "assistant",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]

    conversation_string = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    print(tokenizer.default_chat_template)
    print(conversation_string)


# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     # bnb_4bit_use_double_quant=True,
#     # bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     checkpoint,
#     device_map="auto",
#     quantization_config=quantization_config,
# )
# print(model.get_memory_footprint() / 1024**3)  # in GB
# # generate some text
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# print(generator("Hello, how are you?", max_length=100)[0]["generated_text"])


def test_login():
    from config import settings
    from huggingface_hub import login

    token = settings.hf_token.get_secret_value()
    login(token=token)


if __name__ == "__main__":
    # test_chat_template()
    test_login()
