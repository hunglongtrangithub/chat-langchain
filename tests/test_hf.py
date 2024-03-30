from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# checkpoint = "facebook/opt-125m"
checkpoint = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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
