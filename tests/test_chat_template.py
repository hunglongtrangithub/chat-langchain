from jinja2 import Environment, FileSystemLoader
import os
from transformers import AutoTokenizer

checkpoint = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Create a new Environment object with the FileSystemLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(current_dir))

# Load the template
template = env.get_template("chat_template.j2")

# Your list of messages
messages = [
    # {"role": "system", "content": "Welcome to the chat!"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good, thanks!"},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I'm not sure, let me check..."},
    {"role": "user", "content": "Great!"},
]

# Render the template with the messages
result = template.render(
    messages=messages, bos_token="<s>", eos_token="</s>", raise_exception=Exception
)

# Compare with the output from the tokenizer
tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
chat = tokenizer.decode(tokenized_chat[0])

print(result)
# print(chat)
