# %%

import argparse
import json
import re
from tqdm import tqdm
from openai import OpenAI

# %%

KEYS_TO_TRANSLATE = [
    "questions", "answer", "answer_alias", "new_answer", 
    "new_answer_alias", "single_hops", "new_single_hops",
    "cloze_question"
]

TARGET_LANGUAGES = {
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "ja": "Japanese"
}

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    
    # case_id start and stop
    parser.add_argument("--case_id", nargs=2, type=int, metavar=("START", "STOP"), help="Range of case_ids to process (inclusive START, STOP)", required=True)
    parser.add_argument("--language", nargs="+", choices=TARGET_LANGUAGES.keys(), default=list(TARGET_LANGUAGES.keys()), help="Target languages to translate into (default: all)")
    
    return parser.parse_args()

# %%

def parse_model_json(content: str):
    """Parse JSON from LLM output more robustly than plain json.loads(content).

    Difference from a simple try/except json.loads:
    - Simple parsing fails if the model adds wrapper text or markdown fences.
    - This parser also tries to recover JSON from common wrappers.

    Strategy order:
    1) Parse the full string directly.
    2) Parse JSON inside a fenced block like ```json ... ```.
    3) Parse text between the first "{" and last "}".
    4) Unwrap envelope objects like {"final": "{...}"}.

    Examples:
    - '{"answer": "yes"}' -> {"answer": "yes"}
    - '```json\n{"answer": "yes"}\n```' -> {"answer": "yes"}
    - 'Here is output:\n{"answer": "yes"}\nDone.' -> {"answer": "yes"}
    - '{"final": "{\\"answer\\": \\\"yes\\\"}"}' -> {"answer": "yes"}
    - 'Not JSON at all' -> None
    """
    breakpoint()
    def unwrap_envelope(payload):
        if not isinstance(payload, dict):
            return payload

        has_wrapper_key = False
        for wrapper_key in ("final", "result", "data", "output"):
            if wrapper_key not in payload:
                continue

            has_wrapper_key = True

            wrapped = payload[wrapper_key]
            if isinstance(wrapped, dict):
                return wrapped
            if isinstance(wrapped, str):
                unwrapped = parse_model_json(wrapped)
                if unwrapped is not None:
                    return unwrapped

        if has_wrapper_key:
            return None

        return payload

    if not content:
        return None

    try:
        return unwrap_envelope(json.loads(content))
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
    if code_block_match:
        try:
            return unwrap_envelope(json.loads(code_block_match.group(1)))
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return unwrap_envelope(json.loads(content[start : end + 1]))
        except json.JSONDecodeError:
            return None

    return None


def translate_case(case_data, target_lang_name, lang_code, max_retries=3):
    # 1. Extract only the relevant subset to send to the model
    input_json = {key: case_data[key] for key in KEYS_TO_TRANSLATE if key in case_data}
    
    # 2. Setup System Prompt with Rules
    system_prompt = f"""You are an expert NLP dataset translator. Translate the values of the provided JSON from English into {target_lang_name}.
    
Rules:
1. Output ONLY a valid JSON object.
2. ROOT LEVEL ONLY: Do not wrap your response in parent keys like "final", "translated", or "json". The output must start directly with the same root keys as the input (e.g., {{"question": [...], "answer": "..."}}).
3. Translate all string values, arrays of strings, and nested dictionaries into {target_lang_name}.
4. CRITICAL FOR ALIASES: For 'answer_alias', you MUST provide the aliases, common names, short names, and transliterations as they are naturally used in {target_lang_name}. DO NOT copy English acronyms unless they are the primary way the entity is known.
5. Retain any formatting or placeholders like {{}}."""

    # Call the API (Using JSON mode if the model supports it)
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                
                # OPTIONAL: Add a 1-shot example if the model fails to follow instructions
                {"role": "user", "content": json.dumps({"answer": "United Kingdom", "answer_alias": ["Britain", "UK", "GBR", "Great Britain", "The UK", "🇬🇧"]}, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps({"answer": "イギリス", "answer_alias": ["英国", "グレートブリテン", "連合王国", "イギリス連合王国", "🇬🇧"]}, ensure_ascii=False)},
                
                {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)}
            ],
            temperature=0.3, # Lower temperature for translation
            
            # OPTIONAL: Forces the model to return valid JSON
            # Comment out if the model fails to return valid JSON because it conflicts with the Guided Decoding or the JSON schema enforcer
            # response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        translated_json = parse_model_json(content)
        if translated_json is None:
            tqdm.write(f"Failed to parse JSON for lang_code={lang_code} (attempt {attempt}/{max_retries})")
            continue
        
        return translated_json

    return None

# %%

args = parse_args()

endpoint = args.endpoint
deployment_name = args.model
api_key = args.api_key
start, stop = args.case_id
languages = { k: v for k, v in TARGET_LANGUAGES.items() if k in args.language }

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

# %%

data = json.load(open(args.dataset_path, "r", encoding="utf-8"))

# %%

def should_skip_translate(case_item, suffix: str) -> bool:
    required_fields = [
        f"{key}_{suffix}"
        for key in KEYS_TO_TRANSLATE
    ]
    return all(field in case_item for field in required_fields)

# %%

case_index = {
    item["case_id"]: idx
    for idx, item in enumerate(data)
}
target_case_ids = list(range(start, stop + 1))

missing_case_ids = [case_id for case_id in target_case_ids if case_id not in case_index]
if missing_case_ids:
    raise ValueError(f"case_id(s) not found in dataset: {missing_case_ids}")

jobs = [
    (case_id, language)
    for case_id in target_case_ids
    for language in languages.items()
]

# %%

with tqdm(total=len(jobs), desc="Translating", unit="job") as pbar:
    for case_id, (lang_code, lang_name) in jobs:
        pbar.set_postfix(case_id=case_id, lang_code=lang_code)
        
        case_item = data[case_index[case_id]]
        
        if should_skip_translate(case_item, lang_code):
            tqdm.write(f"Skip case_id={case_id}, lang_code={lang_code} (already exists)")
            pbar.update(1)
            continue
        
        tqdm.write(f"Translating case_id={case_id}, lang_code={lang_code}")
    
        translated_data = translate_case(case_item, lang_name, lang_code)
        
        if translated_data is None:
            tqdm.write(f"Translation failed for case_id={case_id}, lang_code={lang_code}, skipping.")
            pbar.update(1)
            continue
            
        # Append translated keys back to the original case with the language suffix
        updated_keys = 0
        for key in KEYS_TO_TRANSLATE:
            if key not in translated_data:
                tqdm.write(f"Key '{key}' not in translated data for case_id={case_id}, lang_code={lang_code}")
                continue

            new_key_name = f"{key}_{lang_code}"
            case_item[new_key_name] = translated_data[key]
            updated_keys += 1

        if updated_keys == 0:
            tqdm.write(f"No expected keys returned for case_id={case_id}, lang_code={lang_code}; keys={list(translated_data.keys())}; skipping save")
            pbar.update(1)
            continue
    
        with open(args.dataset_path, "w", encoding="utf-8") as f:    
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if updated_keys == len(KEYS_TO_TRANSLATE):
            tqdm.write(f"Saved after case_id={case_id}, lang_code={lang_code} (all keys)")
        else:
            tqdm.write(f"Saved after case_id={case_id}, lang_code={lang_code} (partial {updated_keys}/{len(KEYS_TO_TRANSLATE)} keys)")
        pbar.update(1)

# %%
