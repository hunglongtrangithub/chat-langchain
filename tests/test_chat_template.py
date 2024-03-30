# from jinja2 import Environment, FileSystemLoader
# import os
# from transformers import AutoTokenizer

# checkpoint = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # Create a new Environment object with the FileSystemLoader
# current_dir = os.path.dirname(os.path.abspath(__file__))
# env = Environment(loader=FileSystemLoader(current_dir))

# # Load the template
# template = env.get_template("chat_template.j2")

from huggingface_hub import login
import openai
from jinja2 import Template

CHAT_TEMPLATE_STRING = """{%- for message in messages %}
{% if message.role == 'user' -%}
Human: {{- message.content -}}\n
{%- elif message.role == 'assistant' -%}
AI: {{- message.content -}}\n
{%- else -%}
Unknown Role: {{- message.content -}}\n
{%- endif %}
{%- endfor %}"""
client = openai.OpenAI(api_key="yourmom")
template = Template(CHAT_TEMPLATE_STRING)
messages = [
    {"role": "ur", "content": "What is the capital of France?"},
    {"role": "asant", "content": "The capital of France is Paris."},
]
chat_history = template.render(messages=messages).strip()
print(chat_history)
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a friendly chatbot who always responds in the style of a pirate",
#         },
#         {"role": "user", "content": "An increasing sequence from 1 to 10:"},
#     ],
# )
# Your list of messages
messages = [
    {"role": "system", "content": "Welcome to the chat!"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good, thanks!"},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I'm not sure, let me check..."},
    {"role": "user", "content": "Great!"},
]

# # Render the template with the messages
# result = template.render(
#     messages=messages, bos_token="<s>", eos_token="</s>", raise_exception=Exception
# )

# # Compare with the output from the tokenizer
# tokenized_chat = tokenizer.apply_chat_template(
#     messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
# )
# chat = tokenizer.decode(tokenized_chat[0])

# print(result)
# print(chat)


def format_messages(messages, bos_token, eos_token):
    if messages[0]["role"] == "system":
        loop_messages = messages[1:]
        system_message = messages[0]["content"]
    elif not "<<SYS>>" in messages[0]["content"]:
        loop_messages = messages
        system_message = "You are a helpful, respectful and honest assistant..."
    else:
        loop_messages = messages
        system_message = False

    formatted_messages = []
    for i, message in enumerate(loop_messages):
        if (message["role"] == "user") != (i % 2 == 0):
            raise Exception(
                "Conversation roles must alternate user/assistant/user/assistant/..."
            )
        if i == 0 and system_message != False:
            content = (
                "<<SYS>>\n" + system_message + "\n<</SYS>>\n\n" + message["content"]
            )
        else:
            content = message["content"]

        if message["role"] == "user":
            formatted_messages.append(
                bos_token + "[INST] " + content.strip() + " [/INST]"
            )
        elif message["role"] == "system":
            formatted_messages.append("<<SYS>>\n" + content.strip() + "\n<</SYS>>\n\n")
        elif message["role"] == "assistant":
            formatted_messages.append(" " + content.strip() + " " + eos_token)

    return formatted_messages


# print(format_messages(messages, "<s>", "</s>"))
