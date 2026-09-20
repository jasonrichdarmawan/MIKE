import re

from .types import Message

QWEN_TURN_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant|tool)\s*\n"
    r"(.*?)"
    r"(?:<\|im_end\|>|$)",
    flags=re.DOTALL,
)

def qwen_prompt_to_messages(text: str) -> list[Message]:
    return [
        {
            "role": role,
            "content": content.strip(),
        }
        for role, content in QWEN_TURN_PATTERN.findall(text)
    ]