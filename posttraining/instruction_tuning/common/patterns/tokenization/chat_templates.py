# Standard Library
import typing

# Chat templates for different models and formats
# Note: We added {% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}
# because we want the template to not output eos_token if add_generation_prompt=True

CHAT_TEMPLATES: typing.Dict[str, str] = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}
