# %%

import argparse
import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# %%

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
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to input CSV (two_hop_cf_e2.csv or two_hop_cf_e3.csv)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSON file (created/updated incrementally)")

    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--api_key",  type=str, required=True)

    parser.add_argument("--start", type=int, required=True,
                        help="Row id to start from (inclusive)")
    parser.add_argument("--stop",  type=int, required=True,
                        help="Row id to stop at (inclusive)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of rows per batch; each batch translates all "
                             "target languages so every language gets the same entry count. "
                             "Output JSON is saved after each batch.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel translation workers (threads)")
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
    end = content.rfind("}")
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

# %%

def translate_row(input_data: dict, lang_name: str, lang_code: str, max_retries: int = 3):
    """Call the LLM to translate input_data fields into lang_name.

    Prompt structure mirrors llm_translate.py: same system rules, same JSON
    parsing, same retry/backoff pattern. The one-shot example is adapted for
    two-hops fields (templates with {} and cloze source_prompt).
    """
    system_prompt = f"""You are an expert NLP dataset translator. Translate the values of the provided JSON from English into {lang_name}.

Rules:
1. Output ONLY a valid JSON object.
2. ROOT LEVEL ONLY: Do not wrap your response in parent keys. The output must start directly with the same root keys as the input.
3. Translate all string values and arrays of strings into {lang_name}.
4. CRITICAL FOR ALIASES: For alias fields (e2_aliases, e3_aliases, e2_aliases_new, e3_aliases_new), you MUST provide natural aliases and transliterations in {lang_name}. Do not just copy English acronyms.
5. CRITICAL FOR TEMPLATES: The 'r1_template' and 'r2_template' fields contain a {{}} placeholder marking the entity slot. You MUST preserve {{}} exactly as-is in the translated output — only translate the surrounding text.
6. GRAMMAR & NUANCE: Ensure translations are grammatically correct. Preserve meaning precisely.
7. CRITICAL FOR CLOZE: The 'source_prompt' field is an abruptly incomplete sentence for cloze evaluation via next-token prediction. It MUST cut off exactly before the target answer. DO NOT use "...", "___", or any placeholder. For Chinese or Japanese, rewrite the syntax (e.g., move the subject to enable the answer to fall naturally at the absolute end of the string)."""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},

                    # 1-shot example: shows {} preservation in templates and
                    # cloze-style source_prompt cut-off before the answer.
                    {"role": "user", "content": json.dumps({
                        "r1_template": "the country of citizenship of {}",
                        "r2_template": "the head of state of {}",
                        "source_prompt": "the head of state of the country of citizenship of Fernando Santos is",
                        "e3_label": "United Kingdom",
                        "e3_aliases": ["Britain", "UK", "GBR", "Great Britain", "🇬🇧"],
                    }, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps({
                        "r1_template": "{}の国籍国",
                        "r2_template": "{}の国家元首は",
                        "source_prompt": "フェルナンド・サントスの国籍国の国家元首は",
                        "e3_label": "イギリス",
                        "e3_aliases": ["英国", "グレートブリテン", "連合王国", "イギリス連合王国", "🇬🇧"],
                    }, ensure_ascii=False)},

                    {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
                ],
                temperature=0.3,
            )
        except Exception as e:
            tqdm.write(f"API error lang={lang_code} (attempt {attempt}/{max_retries}): {e}")
            time.sleep(30)
            continue

        content = response.choices[0].message.content
        translated = parse_model_json(content)
        if translated is None:
            tqdm.write(f"Failed to parse JSON lang={lang_code} (attempt {attempt}/{max_retries})")
            continue
        return translated

    return None

# %%

args = parse_args()

endpoint        = args.endpoint
deployment_name = args.model
api_key         = args.api_key
languages       = {k: v for k, v in TARGET_LANGUAGES.items() if k in args.language}

client = OpenAI(base_url=endpoint, api_key=api_key)

# %%

df = pd.read_csv(args.csv_path)
id_to_idx = {int(row["id"]): idx for idx, row in df.iterrows()}

# Load existing output JSON if present (enables resuming interrupted runs)
if os.path.exists(args.output_path):
    existing = json.load(open(args.output_path, encoding="utf-8"))
    output_index = {entry["id"]: entry for entry in existing}
    print(f"Loaded {len(output_index)} existing entries from {args.output_path}")
else:
    output_index = {}
    print(f"Starting fresh output at {args.output_path}")

# Determine target row ids; warn about any ids absent from the CSV
target_ids   = [rid for rid in range(args.start, args.stop + 1) if rid in id_to_idx]
missing_ids  = [rid for rid in range(args.start, args.stop + 1) if rid not in id_to_idx]
if missing_ids:
    preview = missing_ids[:10]
    suffix  = "..." if len(missing_ids) > 10 else ""
    print(f"Skipping {len(missing_ids)} missing row id(s): {preview}{suffix}")

# Seed output_index with base CSV data for any rows not yet in the JSON
for rid in target_ids:
    if rid not in output_index:
        output_index[rid] = row_to_base_dict(df.iloc[id_to_idx[rid]])

# %%

def should_skip(entry: dict, lang_code: str) -> bool:
    """True if source_prompt translation already exists (proxy for full row translation)."""
    return f"source_prompt_{lang_code}" in entry


def _translate_job(rid: int, lang_code: str, lang_name: str):
    """Worker function: translate one (row, language) pair."""
    input_data = row_to_input(output_index[rid])
    result = translate_row(input_data, lang_name, lang_code)
    return rid, lang_code, result

# %%

# Process target_ids in batches.
# Each batch covers batch_size rows × all languages, so every language
# receives the same number of new entries per batch.

for batch_start in range(0, len(target_ids), args.batch_size):
    batch_row_ids = target_ids[batch_start : batch_start + args.batch_size]

    batch_jobs = [
        (rid, lang_code, lang_name)
        for rid in batch_row_ids
        for lang_code, lang_name in languages.items()
        if not should_skip(output_index[rid], lang_code)
    ]

    if not batch_jobs:
        tqdm.write(
            f"Batch {batch_start // args.batch_size + 1} "
            f"(rows {batch_row_ids[0]}-{batch_row_ids[-1]}): all already translated, skipping."
        )
        continue

    tqdm.write(
        f"Batch {batch_start // args.batch_size + 1} "
        f"(rows {batch_row_ids[0]}-{batch_row_ids[-1]}): "
        f"{len(batch_jobs)} jobs ({len(batch_row_ids)} rows × {len(languages)} languages)"
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_translate_job, rid, lang_code, lang_name): (rid, lang_code)
            for rid, lang_code, lang_name in batch_jobs
        }

        with tqdm(total=len(futures), desc=f"Batch {batch_start // args.batch_size + 1}", unit="job") as pbar:
            for future in as_completed(futures):
                rid, lang_code, result = future.result()
                pbar.set_postfix(row_id=rid, lang=lang_code)

                if result is None:
                    tqdm.write(f"Translation failed row_id={rid} lang={lang_code}, skipping.")
                    pbar.update(1)
                    continue

                updated = 0
                for key in KEYS_TO_TRANSLATE:
                    if key not in result:
                        tqdm.write(f"Key '{key}' missing from translation row_id={rid} lang={lang_code}")
                        continue
                    output_index[rid][f"{key}_{lang_code}"] = result[key]
                    updated += 1

                if updated == 0:
                    tqdm.write(
                        f"No expected keys returned row_id={rid} lang={lang_code}; "
                        f"got={list(result.keys())}; skipping save"
                    )
                else:
                    tqdm.write(
                        f"row_id={rid} lang={lang_code}: "
                        f"{updated}/{len(KEYS_TO_TRANSLATE)} keys translated"
                    )
                pbar.update(1)

    # Save after each complete batch (all languages, all rows in batch)
    output_list = [output_index[rid] for rid in sorted(output_index.keys())]
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
    tqdm.write(f"Saved {len(output_list)} entries → {args.output_path}")

# %%
