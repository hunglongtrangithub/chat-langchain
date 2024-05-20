# key: model name on Hugging Face, value: the jinja2 template be set to tokenizer.chat_template
CHAT_TEMPLATES = {
    "meta-llama/Llama-2-7b-chat-hf": """\
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %} 
    {% set system_message = messages[0]['content'] %}
{% elif not '<<SYS>>' in messages[0]['content'] %}
    {% set loop_messages = messages %}
    {% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{%- for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
        {% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + content.strip() + ' [/INST]' -}}
    {% elif message['role'] == 'assistant' %}
        {{- ' '  + content.strip() + ' ' + eos_token -}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{- ' ' -}}
{% endif %}\
""",
    "georgesung/llama2_7b_chat_uncensored": """\
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{%- for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
        {% set content = system_message + '\n\n' + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{- '### HUMAN:' + '\n' + content.strip() + '\n\n' -}}
    {% elif message['role'] == 'assistant' %}
        {{- '### RESPONSE:' + '\n' + content.strip() + '\n\n' -}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{- '### RESPONSE:' + '\n' -}}
{% endif %}\
""",
    "Tap-M/Luna-AI-Llama2-Uncensored": """\
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{%- for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
        {% set content = system_message + '\n\n' + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{- 'USER: ' + content.strip() + '\n' -}}
    {% elif message['role'] == 'assistant' %}
        {{- 'ASSISTANT: ' + content.strip() + '\n' -}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{- 'ASSISTANT: ' -}}
{% endif %}\
""",
    "meta-llama/Meta-Llama-3-8B-Instruct": """\
{% set default_system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}
{% set first_message = messages[0] if messages else None %}
{% if not first_message or first_message['role'] != 'system' %}
    {% set messages = [{'role': 'system', 'content': default_system_message}] + messages %}
{% endif %}
{%- for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 1) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% set header = '<|start_header_id|>' + message['role'] + '<|end_header_id|>' %}
    {% set msg = header + '\n\n' + message['content'] + '<|eot_id|>' %}
    {% if message['role'] == 'system' %}
        {{- '<|begin_of_text|>' + msg -}}
    {% else %}
        {{- msg -}}
    {% endif %}
{% endfor %}\
{% if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
{% endif %}\
""",
}
