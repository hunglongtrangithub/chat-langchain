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
from custom_chat_model import CustomChatModel, CustomModel

# checkpoint = "facebook/opt-125m"
checkpoint = "meta-llama/Llama-2-7b-chat-hf"


def test_chat_model(stream=False):
    model = CustomChatModel(checkpoint)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting? " * 1000,
        },
    ]
    if not stream:
        print(model.invoke(messages))
    else:
        for token in model.stream(messages):
            print(token, end="", flush=True)


def test_model(stream=False):
    model = CustomModel(checkpoint)
    input_text = "How many helicopters can a human eat in one sitting? " * 1000
    if not stream:
        print(model.invoke(input_text))
    else:
        for token in model.stream(input_text):
            print(token, end="", flush=True)


if __name__ == "__main__":
    # test_chat_model()
    # test_chat_model(stream=True)
    # test_model()
    test_model(stream=True)
    print("All tests passed")
