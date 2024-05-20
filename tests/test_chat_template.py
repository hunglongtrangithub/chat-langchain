from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from pathlib import Path
from jinja2 import Template

sys.path.append(str(Path(__file__).resolve().parent.parent))
from chat_templates import CHAT_TEMPLATES


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


def test_llama2_chat_template():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        chat_template=CHAT_TEMPLATES[model_id],
    )
    print(chat)
    # chat_with_default_template = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )
    # print(chat_with_default_template)
    # print(chat.strip() == chat_with_default_template.strip())
    # print(len(chat), len(chat_with_default_template))


def test_uncensored_llama2_chat_template():
    model_id = "georgesung/llama2_7b_chat_uncensored"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATES[model_id],
    )
    chat_with_default_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(chat)
    print(chat_with_default_template)
    print(chat.strip() == chat_with_default_template.strip())
    print(len(chat), len(chat_with_default_template))


def test_llama3_instruct_template():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=CHAT_TEMPLATES[model_id],
    )
    print(chat)


def test_chat_template_string():
    CHAT_TEMPLATE_STRING = """\
{%- for message in messages %}
{% if message.role == 'user' -%}
Human: {{ message.content -}}\n
{%- elif message.role == 'assistant' -%}
AI: {{ message.content -}}\n
{%- else -%}
Unknown Role: {{ message.content -}}\n
{%- endif -%}
{%- endfor -%}\
"""
    template = Template(CHAT_TEMPLATE_STRING)
    rendered = template.render(messages=messages)
    print(rendered.strip())
    print(len(rendered), len(rendered.strip()))


if __name__ == "__main__":
    test_llama2_chat_template()
    # test_chat_template_string()
    # test_uncensored_llama2_chat_template()
    # test_llama3_instruct_template()
