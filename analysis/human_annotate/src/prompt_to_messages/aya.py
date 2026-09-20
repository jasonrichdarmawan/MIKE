import re

from .types import Message

AYA_ROLE_MAP = {
    "SYSTEM": "system",
    "USER": "user",
    "CHATBOT": "assistant",
}

AYA_TURN_PATTERN = re.compile(
    r"<\|START_OF_TURN_TOKEN\|>"
    r"<\|(SYSTEM|USER|CHATBOT)_TOKEN\|>"
    r"(.*?)"
    r"(?:<\|END_OF_TURN_TOKEN\|>|$)",
    flags=re.DOTALL,
)

def ayan_prompt_to_messages(prompt: str) -> list[Message]:
    """Convert an Aya-formatted prompt into chat messages."""
    return [
        {
            "role": AYA_ROLE_MAP[token_role],
            "content": content.strip(),
        }
        for token_role, content in AYA_TURN_PATTERN.findall(prompt)
    ]