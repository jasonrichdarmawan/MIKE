# %%

import argparse
import ast
import asyncio
import json
import os
import re

import aiofiles
import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

# %%

ENDPOINT    = "https://algoverse-2026-02-resource.openai.azure.com/openai/v1/"
API_KEY     = "9W4R1LKl3FOZVrjGfuQlYf16aaX196wTZnGjQz8LHhRNEiWhmMw4JQQJ99CCACHYHv6XJ3w3AAAAACOGJhQm"

TARGET_LANGUAGES = {
    "fr": "French",
    "zh": "Chinese",
    "ja": "Japanese",
}

# Fields to translate from each CSV row.
# Covers single-hop templates (for bridge retrieval & composition isolation)
# and the multi-hop source prompt, plus all answer entity labels/aliases
# needed for evaluation at each stage.
KEYS_TO_TRANSLATE = [
    "source_prompt",    # multi-hop cloze prompt  (e1 → e2 → e3)
    "r1_template",      # single-hop template 1   (e1 → e2, bridge entity retrieval)
    "r2_template",      # single-hop template 2   (e2 → e3, composition step)
    "e2_label",         # bridge entity label
    "e2_aliases",       # bridge entity aliases
    "e3_label",         # final answer label
    "e3_aliases",       # final answer aliases
    "e2_label_new",     # new bridge entity label (post-edit)
    "e2_aliases_new",   # new bridge entity aliases (post-edit)
    "e3_label_new",     # new answer label (post-edit)
    "e3_aliases_new",   # new answer aliases (post-edit)
]

# Alias fields stored as string repr of Python lists in the CSV
ALIAS_FIELDS = {"e2_aliases", "e3_aliases", "e2_aliases_new", "e3_aliases_new"}

# %%

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Translate two-hops CounterFact CSV into multilingual JSON. "
            "Reads two_hop_cf_e2.csv or two_hop_cf_e3.csv and writes a JSON "
            "file with translated fields appended as language-suffixed keys."
        )
    )
    parser.add_argument("--csv_path",    type=str, required=True,
                        help="Path to input CSV (two_hop_cf_e2.csv or two_hop_cf_e3.csv)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSON file (created/updated incrementally)")

    parser.add_argument("--model",    type=str, required=True,
                        help="Azure deployment name (e.g. gpt-4o)")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT)
    parser.add_argument("--api_key",  type=str, default=API_KEY)

    parser.add_argument("--start", type=int, required=True,
                        help="Row id to start from (inclusive)")
    parser.add_argument("--stop",  type=int, required=True,
                        help="Row id to stop at (inclusive)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of rows per batch. Each batch translates all "
                             "target languages so every language gets the same entry "
                             "count. Output JSON is saved after each batch.")
    parser.add_argument("--max_concurrency", type=int, default=8,
                        help="Max simultaneous API calls (tune to your Azure RPM quota).")
    parser.add_argument("--language", nargs="+", choices=TARGET_LANGUAGES.keys(),
                        default=list(TARGET_LANGUAGES.keys()),
                        help="Target languages to translate into (default: all)")
    return parser.parse_args()

# %%

def parse_model_json(content: str):
    """Parse JSON from LLM output more robustly than plain json.loads(content).

    Identical to llm_translate.py for consistency across translation scripts.

    Strategy order:
    1) Parse the full string directly.
    2) Parse JSON inside a fenced block like ```json ... ```.
    3) Parse text between the first "{" and last "}".
    4) Unwrap envelope objects like {"final": "{...}"}.
    """
    def unwrap_envelope(payload):
        if not isinstance(payload, dict):
            return payload
        for wrapper_key in ("final", "result", "data", "output"):
            if wrapper_key not in payload:
                continue
            wrapped = payload[wrapper_key]
            if isinstance(wrapped, dict):
                return wrapped
            if isinstance(wrapped, str):
                unwrapped = parse_model_json(wrapped)
                if unwrapped is not None:
                    return unwrapped
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
    end   = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return unwrap_envelope(json.loads(content[start:end + 1]))
        except json.JSONDecodeError:
            return None

    return None

# %%

def parse_aliases(value) -> list:
    """Convert a CSV alias field (string repr of a list) into an actual list."""
    if isinstance(value, list):
        return value
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else [str(result)]
    except (ValueError, SyntaxError):
        return [str(value)]


def row_to_input(entry: dict) -> dict:
    """Build the translation input dict from an output_index entry.

    Skips fields that are absent or empty so the LLM payload stays minimal.
    Alias fields are already parsed lists at this point (set during row_to_base_dict).
    """
    input_data = {}
    for key in KEYS_TO_TRANSLATE:
        val = entry.get(key)
        if val is None or val == "":
            continue
        if isinstance(val, list) and len(val) == 0:
            continue
        input_data[key] = val
    return input_data


def row_to_base_dict(row) -> dict:
    """Convert a pandas Series (CSV row) to a clean base dict for JSON output.

    Alias fields are parsed from their CSV string-repr into proper Python lists
    so the output JSON contains real arrays instead of escaped strings.
    """
    d = {}
    for col in row.index:
        val = row[col]
        if isinstance(val, float) and pd.isna(val):
            d[col] = None
        elif col in ALIAS_FIELDS:
            d[col] = parse_aliases(val)
        else:
            try:
                d[col] = val.item()   # convert numpy int/float → Python native
            except AttributeError:
                d[col] = val
    return d


def should_skip(entry: dict, lang_code: str) -> bool:
    """True if source_prompt translation already exists (proxy for full row translation)."""
    return f"source_prompt_{lang_code}" in entry

# %%

def build_system_prompt(lang_name: str) -> str:
    return f"""You are an expert NLP dataset translator. Translate the values of the provided JSON from English into {lang_name}.

Rules:
1. Output ONLY a valid JSON object.
2. ROOT LEVEL ONLY: Do not wrap your response in parent keys. The output must start directly with the same root keys as the input.
3. Translate all string values and arrays of strings into {lang_name}.
4. CRITICAL FOR ALIASES: For alias fields (e2_aliases, e3_aliases, e2_aliases_new, e3_aliases_new), you MUST provide natural aliases and transliterations in {lang_name}. Do not just copy English acronyms.
5. CRITICAL FOR TEMPLATES: The 'r1_template' and 'r2_template' fields contain a {{}} placeholder marking the entity slot. You MUST preserve {{}} exactly as-is in the translated output — only translate the surrounding text.
6. GRAMMAR & NUANCE: Ensure translations are grammatically correct. Preserve meaning precisely.
7. CRITICAL FOR CLOZE: The 'source_prompt' field is an abruptly incomplete sentence for cloze evaluation via next-token prediction. It MUST cut off exactly before the target answer. DO NOT use "...", "___", or any placeholder. For Chinese or Japanese, rewrite the syntax (e.g., move the subject to enable the answer to fall naturally at the absolute end of the string)."""


# 1-shot example shared across all calls: shows {} preservation in templates
# and cloze-style source_prompt cut-off before the answer.
FEW_SHOT_USER = json.dumps({
    "r1_template": "the country of citizenship of {}",
    "r2_template": "the head of state of {}",
    "source_prompt": "the head of state of the country of citizenship of Fernando Santos is",
    "e3_label": "United Kingdom",
    "e3_aliases": ["Britain", "UK", "GBR", "Great Britain", "🇬🇧"],
}, ensure_ascii=False)

FEW_SHOT_ASSISTANT = json.dumps({
    "r1_template": "{}の国籍国",
    "r2_template": "{}の国家元首は",
    "source_prompt": "フェルナンド・サントスの国籍国の国家元首は",
    "e3_label": "イギリス",
    "e3_aliases": ["英国", "グレートブリテン", "連合王国", "イギリス連合王国", "🇬🇧"],
}, ensure_ascii=False)

# %%

@retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
)
async def _api_call(client: AsyncOpenAI, deployment_name: str,
                    system_prompt: str, input_data: dict) -> str:
    """Single async API call with tenacity retry on rate-limit / server errors."""
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user",      "content": json.dumps(input_data, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


async def translate_row(client: AsyncOpenAI, deployment_name: str,
                        semaphore: asyncio.Semaphore,
                        input_data: dict, lang_name: str, lang_code: str):
    """Translate one row into one language, respecting the concurrency semaphore."""
    system_prompt = build_system_prompt(lang_name)
    async with semaphore:
        try:
            content = await _api_call(client, deployment_name, system_prompt, input_data)
        except Exception as e:
            tqdm.write(f"API error row lang={lang_code}: {e}")
            return None

    translated = parse_model_json(content)
    if translated is None:
        tqdm.write(f"Failed to parse JSON lang={lang_code}")
    return translated


async def save_json(output_path: str, output_index: dict):
    """Async write of the full output JSON after each batch."""
    output_list = [output_index[rid] for rid in sorted(output_index.keys())]
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(output_list, ensure_ascii=False, indent=2))
    return len(output_list)

# %%

async def main():
    args = parse_args()

    deployment_name = args.model
    languages       = {k: v for k, v in TARGET_LANGUAGES.items() if k in args.language}
    semaphore       = asyncio.Semaphore(args.max_concurrency)

    client = AsyncOpenAI(base_url=args.endpoint, api_key=args.api_key)

    # ── load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv_path)
    id_to_idx = {int(row["id"]): idx for idx, row in df.iterrows()}

    # ── load or initialise output JSON (enables resuming interrupted runs) ────
    if os.path.exists(args.output_path):
        existing     = json.load(open(args.output_path, encoding="utf-8"))
        output_index = {entry["id"]: entry for entry in existing}
        print(f"Loaded {len(output_index)} existing entries from {args.output_path}")
    else:
        output_index = {}
        print(f"Starting fresh output at {args.output_path}")

    # ── determine target row ids ──────────────────────────────────────────────
    target_ids  = [rid for rid in range(args.start, args.stop + 1) if rid in id_to_idx]
    missing_ids = [rid for rid in range(args.start, args.stop + 1) if rid not in id_to_idx]
    if missing_ids:
        preview = missing_ids[:10]
        suffix  = "..." if len(missing_ids) > 10 else ""
        print(f"Skipping {len(missing_ids)} missing row id(s): {preview}{suffix}")

    # seed output_index with base CSV data for rows not yet in the JSON
    for rid in target_ids:
        if rid not in output_index:
            output_index[rid] = row_to_base_dict(df.iloc[id_to_idx[rid]])

    # ── process in batches ────────────────────────────────────────────────────
    # Each batch = batch_size rows × all languages → same entry count per language.
    total_batches = (len(target_ids) + args.batch_size - 1) // args.batch_size

    for batch_num, batch_start in enumerate(range(0, len(target_ids), args.batch_size), 1):
        batch_row_ids = target_ids[batch_start : batch_start + args.batch_size]

        batch_jobs = [
            (rid, lang_code, lang_name)
            for rid in batch_row_ids
            for lang_code, lang_name in languages.items()
            if not should_skip(output_index[rid], lang_code)
        ]

        if not batch_jobs:
            print(f"Batch {batch_num}/{total_batches} "
                  f"(rows {batch_row_ids[0]}-{batch_row_ids[-1]}): already translated, skipping.")
            continue

        print(f"Batch {batch_num}/{total_batches} "
              f"(rows {batch_row_ids[0]}-{batch_row_ids[-1]}): "
              f"{len(batch_jobs)} jobs  "
              f"({len(batch_row_ids)} rows × {len(languages)} languages)")

        # fire all jobs concurrently; semaphore caps simultaneous API calls
        async def run_job(rid, lang_code, lang_name):
            input_data = row_to_input(output_index[rid])
            result     = await translate_row(
                client, deployment_name, semaphore, input_data, lang_name, lang_code
            )
            return rid, lang_code, result

        tasks   = [asyncio.create_task(run_job(r, lc, ln)) for r, lc, ln in batch_jobs]
        results = await atqdm.gather(*tasks, desc=f"Batch {batch_num}/{total_batches}", unit="job")

        for rid, lang_code, result in results:
            if result is None:
                tqdm.write(f"Translation failed row_id={rid} lang={lang_code}, skipping.")
                continue

            updated = 0
            for key in KEYS_TO_TRANSLATE:
                if key not in result:
                    tqdm.write(f"Key '{key}' missing row_id={rid} lang={lang_code}")
                    continue
                output_index[rid][f"{key}_{lang_code}"] = result[key]
                updated += 1

            if updated == 0:
                tqdm.write(f"No expected keys row_id={rid} lang={lang_code}; "
                           f"got={list(result.keys())}")
            else:
                tqdm.write(f"row_id={rid} lang={lang_code}: "
                           f"{updated}/{len(KEYS_TO_TRANSLATE)} keys")

        # save after each complete batch
        n_saved = await save_json(args.output_path, output_index)
        tqdm.write(f"Saved {n_saved} entries → {args.output_path}")

    await client.close()

# %%

if __name__ == "__main__":
    asyncio.run(main())

# %%
