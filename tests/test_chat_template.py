from jinja2 import Environment, FileSystemLoader
import os

# Create a new Environment object with the FileSystemLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(current_dir))

# Load the template
template = env.get_template("chat_template.j2")

# Your list of messages
messages = [
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
    messages=messages, bos_token="[BOS]", eos_token="[EOS]", raise_exception=Exception
)

print(result)
