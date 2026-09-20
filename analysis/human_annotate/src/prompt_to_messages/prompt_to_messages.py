from .types import Message

def prompt_to_messages(prompt: str, model_name: str) -> list[Message]:
    match model_name:
        case "CohereLabs/aya-expanse-8b":
            from .aya import ayan_prompt_to_messages
            return ayan_prompt_to_messages(prompt)
        case "Qwen/Qwen3-4B-Instruct-2507" | "Qwen/Qwen2.5-7B-Instruct":
            from .qwen import qwen_prompt_to_messages
            return qwen_prompt_to_messages(prompt)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")