# %%

import argparse
import ast
import asyncio
import json
import os
import re

import aiofiles
import httpx
import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import (retry, stop_after_attempt, wait_random_exponential,
                      retry_if_exception, before_sleep_log)
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
import logging

API_TIMEOUT_SECONDS = 120  # abort any single API call after this long
MAX_COMPLETION_PASSES = 20  # retry passes after main batches until all rows done
INTER_PASS_SLEEP_SECONDS = 60  # wait between completion passes

_logger = logging.getLogger(__name__)


def _load_env_file(env_path: str) -> None:
    """Load KEY=VALUE pairs from a local env file into os.environ."""
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_load_env_file(os.path.join(PROJECT_ROOT, "gpt-oss.env"))


def _is_retryable(exc: BaseException) -> bool:
    """Retry only on rate-limit or transient server errors, not bad requests."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, asyncio.TimeoutError):
        return True
    return False


def _before_sleep(retry_state):
    exc = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    # Respect Retry-After header from Azure if present
    if isinstance(exc, (RateLimitError, APIStatusError)):
        resp = getattr(exc, "response", None)
        if resp is not None:
            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            if retry_after:
                tqdm.write(f"Rate limited — Retry-After={retry_after}s "
                           f"(attempt {retry_state.attempt_number})")
                return
    tqdm.write(f"Retrying after error (attempt {retry_state.attempt_number}, "
               f"wait {wait:.1f}s): {type(exc).__name__}: {exc}")

# %%

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")

TARGET_LANGUAGES = {
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "hi": "Hindi",
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
    parser.add_argument("--endpoint",   type=str, default=ENDPOINT)
    parser.add_argument("--api_key",    type=str, default=API_KEY)

    parser.add_argument("--start", type=int, default=None,
                        help="Row id to start from (inclusive). Defaults to first id in CSV.")
    parser.add_argument("--stop",  type=int, default=None,
                        help="Row id to stop at (inclusive). Defaults to last id in CSV.")
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
    """True if the language block already exists for this entry."""
    return lang_code in entry and "source_prompt" in entry[lang_code]

# %%

def build_system_prompt(lang_name: str, lang_code: str) -> str:
    lang_extra = ""
    if lang_code == "hi":
        lang_extra = """

HINDI-SPECIFIC (highest priority):
- Hindi is SOV. The verb "है" must NEVER appear at the end of source_prompt — it pre-answers the cloze.
- Restructure source_prompt using a genitive/possessive form ending with "=" so the answer follows naturally. Example: "फ्रांस के लुई IX की माँ के पिता =" not "फ्रांस के लुई IX की माँ के पिता हैं".
- The "=" is the fill-in marker. Do NOT add है/हैं/का नाम or any verb after "=".
- NEVER use question words: कौन, क्या, कहाँ, कब. Always rephrase as a statement prefix.
- r1_template and r2_template must also use genitive/possessive form with {} preserved."""
    elif lang_code == "zh":
        lang_extra = """

CHINESE-SPECIFIC:
- Use topic-comment or genitive structure so the answer falls at the very end after "是".
- PREFERRED pattern: "X的[relation]是" (e.g. "路易九世的母亲的父亲是").
- Do NOT use question words: 谁, 什么, 哪个, 哪位.
- r1_template and r2_template must use "{}的[relation]" pattern with {} preserved."""
    elif lang_code == "ja":
        lang_extra = """

JAPANESE-SPECIFIC:
- Japanese is SOV. The verb always comes last, so "X is Y's mother" becomes "XのYの母は" (X's Y's mother is) — answer follows the verb naturally.
- Use "{}の[relation]は" pattern for r1_template and r2_template, preserving {} exactly.
- source_prompt must end with "は" (topic marker) or "が" before the answer — never include the answer itself.
- Do NOT use question words: 誰 (who), 何 (what), どの (which). Keep as a statement prefix."""
    elif lang_code == "de":
        lang_extra = """

GERMAN-SPECIFIC:
- German V2 word order: the finite verb (ist, war) is the second constituent in main clauses.
- source_prompt ends naturally with the verb before the answer (e.g. "Der Vater der Mutter von Ludwig IX. von Frankreich ist").
- r1_template and r2_template must preserve {} exactly."""

    return f"""You are an expert NLP dataset translator. Translate the values of the provided JSON from English into {lang_name}.

PRIORITY ORDER:
1. STRUCTURE — preserve all JSON keys and array types exactly.
2. CLOZE POSITION — source_prompt must end immediately before the target answer (next-token prediction format). Do NOT include the answer.
3. GRAMMAR — ensure natural, correct {lang_name}.

RULES:
1. Output ONLY a valid JSON object. No prose, no markdown fences.
2. ROOT LEVEL ONLY: start directly with the same root keys as the input.
3. Translate all string values and arrays of strings into {lang_name}.
4. ALIASES: provide natural aliases and transliterations. Do not copy English acronyms unchanged if native alternatives exist.
5. TEMPLATES: r1_template and r2_template contain a {{}} placeholder — preserve {{}} exactly, only translate the surrounding text.
6. CLOZE: source_prompt is an abruptly incomplete prefix for cloze evaluation. Do NOT use "...", "___", or any filler. Cut off right before the answer.
7. Do NOT turn source_prompt into a natural question (avoid who/what/which at the start).{lang_extra}"""


# Shared few-shot input used for all languages
_FEW_SHOT_INPUT = {
    "source_prompt": "the father of the mother of Louis IX of France is",
    "r1_template": "the mother of {}",
    "r2_template": "the father of {}",
    "e2_label": "Blanche of Castile",
    "e2_aliases": ["Blanche"],
    "e3_label": "Alfonso VIII of Castile",
    "e3_aliases": ["Alfonso VIII"],
    "e2_label_new": "Stanley Ann Dunham",
    "e2_aliases_new": ["Ann Obama", "Ann Soetoro"],
    "e3_label_new": "Stanley Armour Dunham",
    "e3_aliases_new": ["Stanley Dunham"],
}

FEW_SHOT_USER = json.dumps(_FEW_SHOT_INPUT, ensure_ascii=False)

FEW_SHOT_ASSISTANT_FR = json.dumps({
    "source_prompt": "le père de la mère de Louis IX de France est",
    "r1_template": "la mère de {}",
    "r2_template": "le père de {}",
    "e2_label": "Blanche de Castille",
    "e2_aliases": ["Blanche"],
    "e3_label": "Alphonse VIII de Castille",
    "e3_aliases": ["Alphonse VIII"],
    "e2_label_new": "Stanley Ann Dunham",
    "e2_aliases_new": ["Ann Obama", "Ann Soetoro"],
    "e3_label_new": "Stanley Armour Dunham",
    "e3_aliases_new": ["Stanley Dunham"],
}, ensure_ascii=False)

FEW_SHOT_ASSISTANT_DE = json.dumps({
    "source_prompt": "der Vater der Mutter von Ludwig IX. von Frankreich ist",
    "r1_template": "die Mutter von {}",
    "r2_template": "der Vater von {}",
    "e2_label": "Blanche von Kastilien",
    "e2_aliases": ["Blanche"],
    "e3_label": "Alfonso VIII. von Kastilien",
    "e3_aliases": ["Alfonso VIII"],
    "e2_label_new": "Stanley Ann Dunham",
    "e2_aliases_new": ["Ann Obama", "Ann Soetoro"],
    "e3_label_new": "Stanley Armour Dunham",
    "e3_aliases_new": ["Stanley Dunham"],
}, ensure_ascii=False)

FEW_SHOT_ASSISTANT_JA = json.dumps({
    "source_prompt": "フランスのルイ9世の母親の父親は",
    "r1_template": "{}の母親",
    "r2_template": "{}の父親",
    "e2_label": "カスティーリャのブランシュ",
    "e2_aliases": ["ブランシュ"],
    "e3_label": "カスティーリャのアルフォンソ8世",
    "e3_aliases": ["アルフォンソ8世"],
    "e2_label_new": "スタンリー・アン・ダンハム",
    "e2_aliases_new": ["アン・オバマ", "アン・ソエトロ"],
    "e3_label_new": "スタンリー・アーマー・ダンハム",
    "e3_aliases_new": ["スタンリー・ダンハム"],
}, ensure_ascii=False)

FEW_SHOT_ASSISTANT_ZH = json.dumps({
    "source_prompt": "法国路易九世的母亲的父亲是",
    "r1_template": "{}的母亲",
    "r2_template": "{}的父亲",
    "e2_label": "卡斯蒂利亚的布朗什",
    "e2_aliases": ["布朗什"],
    "e3_label": "卡斯蒂利亚的阿方索八世",
    "e3_aliases": ["阿方索八世"],
    "e2_label_new": "斯坦利·安·邓纳姆",
    "e2_aliases_new": ["安·奥巴马", "安·索托罗"],
    "e3_label_new": "斯坦利·阿莫尔·邓纳姆",
    "e3_aliases_new": ["斯坦利·邓纳姆"],
}, ensure_ascii=False)

FEW_SHOT_ASSISTANT_HI = json.dumps({
    "source_prompt": "फ्रांस के लुई IX की माँ के पिता =",
    "r1_template": "{} की माँ",
    "r2_template": "{} के पिता",
    "e2_label": "ब्लैंच ऑफ़ कास्तिल",
    "e2_aliases": ["ब्लैंच"],
    "e3_label": "अल्फोंसो VIII ऑफ़ कास्तिल",
    "e3_aliases": ["अल्फोंसो VIII"],
    "e2_label_new": "स्टैनली ऐन डनहैम",
    "e2_aliases_new": ["ऐन ओबामा", "ऐन सोएतोरो"],
    "e3_label_new": "स्टैनली आर्मर डनहैम",
    "e3_aliases_new": ["स्टैनली डनहैम"],
}, ensure_ascii=False)

_FEW_SHOT_ASSISTANT = {
    "fr": FEW_SHOT_ASSISTANT_FR,
    "de": FEW_SHOT_ASSISTANT_DE,
    "zh": FEW_SHOT_ASSISTANT_ZH,
    "ja": FEW_SHOT_ASSISTANT_JA,
    "hi": FEW_SHOT_ASSISTANT_HI,
}

# %%

@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(min=5, max=180),
    stop=stop_after_attempt(40),
    before_sleep=_before_sleep,
    reraise=True,
)
async def _api_call(client: AsyncOpenAI, deployment_name: str,
                    system_prompt: str, few_shot_assistant: str,
                    input_data: dict) -> str:
    """Single async API call — retries indefinitely on rate-limit / server errors."""
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": FEW_SHOT_USER},
                {"role": "assistant", "content": few_shot_assistant},
                {"role": "user",      "content": json.dumps(input_data, ensure_ascii=False)},
            ],
            temperature=0.3,
        ),
        timeout=API_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content


async def translate_row(client: AsyncOpenAI, deployment_name: str,
                        semaphore: asyncio.Semaphore,
                        input_data: dict, lang_name: str, lang_code: str,
                        max_parse_retries: int = 5):
    """Translate one row into one language, respecting the concurrency semaphore.

    Retries up to max_parse_retries times on unparseable JSON or missing keys.
    """
    system_prompt      = build_system_prompt(lang_name, lang_code)
    few_shot_assistant = _FEW_SHOT_ASSISTANT[lang_code]

    for attempt in range(1, max_parse_retries + 1):
        async with semaphore:
            try:
                content = await _api_call(client, deployment_name, system_prompt,
                                          few_shot_assistant, input_data)
            except Exception as e:
                tqdm.write(f"API error lang={lang_code} (attempt {attempt}/{max_parse_retries}): {e}")
                return None

        translated = parse_model_json(content)
        if translated is None:
            tqdm.write(f"Failed to parse JSON lang={lang_code} (attempt {attempt}/{max_parse_retries})")
            continue

        # Require at least source_prompt in the response
        if "source_prompt" not in translated:
            tqdm.write(f"source_prompt missing lang={lang_code} (attempt {attempt}/{max_parse_retries})")
            continue

        return translated

    return None


async def save_json(output_path: str, output_index: dict, lock: asyncio.Lock):
    """Async write of the full output JSON, serialised via lock to prevent corruption."""
    async with lock:
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
    save_lock       = asyncio.Lock()   # prevents concurrent writes corrupting the JSON

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
    all_csv_ids = sorted(id_to_idx.keys())
    start = args.start if args.start is not None else all_csv_ids[0]
    stop  = args.stop  if args.stop  is not None else all_csv_ids[-1]
    target_ids = [rid for rid in all_csv_ids if start <= rid <= stop]
    print(f"Target rows: {len(target_ids)} (ids {start}–{stop})")

    # seed output_index with base CSV data for rows not yet in the JSON
    for rid in target_ids:
        if rid not in output_index:
            base = row_to_base_dict(df.iloc[id_to_idx[rid]])
            output_index[rid] = base
        # always (re)seed the English block from CSV fields
        if "en" not in output_index[rid]:
            output_index[rid]["en"] = row_to_input(output_index[rid])

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

        # fire all jobs concurrently; semaphore caps simultaneous API calls;
        # each job saves immediately on success — progress survives Ctrl+C
        async def run_job(rid, lang_code, lang_name):
            input_data = row_to_input(output_index[rid])
            result     = await translate_row(
                client, deployment_name, semaphore, input_data, lang_name, lang_code
            )
            if result is None:
                tqdm.write(f"Translation failed row_id={rid} lang={lang_code}, skipping.")
                return

            lang_block = {}
            for key in KEYS_TO_TRANSLATE:
                if key not in result:
                    tqdm.write(f"Key '{key}' missing row_id={rid} lang={lang_code}")
                    continue
                lang_block[key] = result[key]

            if not lang_block:
                tqdm.write(f"No expected keys row_id={rid} lang={lang_code}; "
                           f"got={list(result.keys())}")
                return

            output_index[rid][lang_code] = lang_block
            tqdm.write(f"row_id={rid} lang={lang_code}: {len(lang_block)}/{len(KEYS_TO_TRANSLATE)} keys")
            await save_json(args.output_path, output_index, save_lock)

        tasks = [asyncio.create_task(run_job(r, lc, ln)) for r, lc, ln in batch_jobs]
        await atqdm.gather(*tasks, desc=f"Batch {batch_num}/{total_batches}", unit="job")
        tqdm.write(f"Batch {batch_num}/{total_batches} complete.")

    # ── completion loop: keep retrying until every row×language is done ───────
    def get_missing():
        return [
            (rid, lang_code, lang_name)
            for rid in target_ids
            for lang_code, lang_name in languages.items()
            if not should_skip(output_index[rid], lang_code)
        ]

    for pass_num in range(1, MAX_COMPLETION_PASSES + 1):
        missing_jobs = get_missing()
        if not missing_jobs:
            print(f"All {len(target_ids) * len(languages)} jobs complete.")
            break

        print(f"Completion pass {pass_num}/{MAX_COMPLETION_PASSES}: "
              f"{len(missing_jobs)} missing job(s) — retrying "
              f"(sleeping {INTER_PASS_SLEEP_SECONDS}s first to let rate limits recover)...")
        await asyncio.sleep(INTER_PASS_SLEEP_SECONDS)

        tasks = [asyncio.create_task(run_job(r, lc, ln)) for r, lc, ln in missing_jobs]
        await atqdm.gather(*tasks, desc=f"Completion pass {pass_num}", unit="job")
    else:
        still_missing = [(rid, lc) for rid, lc, _ in get_missing()]
        tqdm.write(f"WARNING: {len(still_missing)} job(s) still untranslated after "
                   f"{MAX_COMPLETION_PASSES} passes: {still_missing[:20]}")

    await client.close()

# %%

if __name__ == "__main__":
    asyncio.run(main())

# %%
