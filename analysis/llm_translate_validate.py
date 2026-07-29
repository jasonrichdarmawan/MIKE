"""Embed resumable LLM translation-audit judgments in an MQuAKE dataset.

This is an automated audit, not bilingual human validation. Each audited text
unit receives ``*_translation_judgement`` and ``*_translation_judgement_reason``
fields in the dataset itself. Re-running the script skips a case-language pair
only when every expected judgment has already been stored.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


TARGET_LANGUAGES = {"zh": "Chinese", "de": "German", "fr": "French", "ja": "Japanese"}
VALID_LABELS = {"accepted", "minor_issue", "major_issue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed resumable LLM translation audits in a dataset.")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument(
        "--case-id", nargs=2, type=int, metavar=("START", "STOP"), required=True,
        help="Inclusive case_id range to audit. Split ranges permit practical resumable runs.",
    )
    parser.add_argument(
        "--languages", nargs="+", choices=tuple(TARGET_LANGUAGES),
        default=tuple(TARGET_LANGUAGES),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Maximum concurrent API requests (default: 8). Lower this if the "
            "endpoint rate-limits requests."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-audit completed case-language pairs instead of skipping them.",
    )
    return parser.parse_args()


def parse_model_json(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    candidates = [content]
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        candidates.append(content[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            for wrapper_key in ("final", "result", "data", "output"):
                wrapped = value.get(wrapper_key)
                if isinstance(wrapped, dict):
                    return wrapped
                if isinstance(wrapped, str):
                    return parse_model_json(wrapped)
            return value
    return None


def atomic_write_json(path: Path, data: Any) -> None:
    """Avoid corrupting the dataset if an API call or process stops unexpectedly."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def build_units(case: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    """Return all audited strings and their stable IDs/locations in one case."""
    suffix = f"_{language}"
    units: List[Dict[str, Any]] = []

    def add(unit_id: str, prompt_type: str, source: Any, translation: Any, container: Dict[str, Any], key: str) -> None:
        units.append({
            "id": unit_id,
            "prompt_type": prompt_type,
            "source": source,
            "translation": translation,
            "container": container,
            "key": key,
        })

    for index, (source, translation) in enumerate(zip(case["questions"], case[f"questions{suffix}"])):
        add(f"multi_hop_question_{index}", "multi_hop_question", source, translation, case, f"questions{suffix}_translation_judgements")
    add("multi_hop_cloze", "multi_hop_cloze", case["cloze_question"], case[f"cloze_question{suffix}"], case, f"cloze_question{suffix}")

    for field_name in ("answer", "answer_alias", "new_answer", "new_answer_alias"):
        prompt_type = "answer_alias" if "alias" in field_name else "answer"
        add(field_name, prompt_type, case[field_name], case[f"{field_name}{suffix}"], case, f"{field_name}{suffix}")

    for hop_group in ("single_hops", "new_single_hops"):
        translated_group = case[f"{hop_group}{suffix}"]
        for index, (source_hop, translated_hop) in enumerate(zip(case[hop_group], translated_group)):
            prefix = f"{hop_group}_{index}"
            for source_key, prompt_type in (("question", "single_hop_question"), ("cloze", "single_hop_cloze"), ("answer", "answer"), ("answer_alias", "answer_alias")):
                add(
                    f"{prefix}_{source_key}", prompt_type, source_hop[source_key], translated_hop[source_key],
                    translated_hop, source_key,
                )
    return units


def validation_key(unit: Dict[str, Any]) -> str:
    """Field that stores this unit's label; questions are represented as a list."""
    if unit["id"].startswith("multi_hop_question_"):
        return "translation_judgement"
    return f"{unit['key']}_translation_judgement"


def reason_key(unit: Dict[str, Any]) -> str:
    if unit["id"].startswith("multi_hop_question_"):
        return "translation_judgement_reason"
    return f"{unit['key']}_translation_judgement_reason"


def is_complete(units: List[Dict[str, Any]]) -> bool:
    return not incomplete_units(units)


def incomplete_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only units without a valid stored judgment.

    This permits a rerun after a partial API response to audit only the missing
    text units instead of overwriting completed judgments for the whole case.
    """
    missing: List[Dict[str, Any]] = []
    for unit in units:
        if unit["id"].startswith("multi_hop_question_"):
            entries = unit["container"].get(unit["key"], [])
            index = int(unit["id"].rsplit("_", 1)[1])
            if not isinstance(entries, list) or len(entries) <= index or entries[index].get("label") not in VALID_LABELS:
                missing.append(unit)
        elif unit["container"].get(validation_key(unit)) not in VALID_LABELS:
            missing.append(unit)
    return missing


def audit_prompt(language_name: str, units: List[Dict[str, Any]]) -> str:
    visible_units = [
        {key: unit[key] for key in ("id", "prompt_type", "source", "translation")}
        for unit in units
    ]
    return f"""You audit English-to-{language_name} translations used in a knowledge-editing benchmark.
For every unit, assess faithfulness, grammar, factual entity preservation, and answer/alias usability. For cloze prompts, verify that the translation remains abruptly incomplete immediately before the answer, so next-token evaluation can produce the answer. Do not assess whether the underlying fact is true.

For answer-alias lists, DO NOT require a one-to-one translation or the same number of entries as the English list. Multiple English aliases may legitimately map to the same target-language form; redundant aliases may be omitted; and target-language transliterations may be added. Mark a major issue only when an alias is materially incorrect, refers to a different entity, or the list lacks usable coverage of the answer. Do not describe an alias as a source mapping unless that source alias actually appears in the supplied English list.

Labels:
- accepted: no material problem;
- minor_issue: usable but a localized grammar, style, transliteration, number, or alias issue;
- major_issue: changed meaning, incorrect/missing entity or answer, malformed text, or a cloze prompt that no longer supports next-token prediction of its answer.

Return ONLY JSON exactly in this form:
{{"judgments": [{{"id": "unit id", "label": "accepted|minor_issue|major_issue", "reason": "brief reason"}}]}}
Return exactly one judgment for every supplied ID.

Units:
{json.dumps(visible_units, ensure_ascii=False)}"""


def normalize_response(payload: Optional[Dict[str, Any]], expected_ids: set[str]) -> Optional[Dict[str, Dict[str, str]]]:
    if not payload or not isinstance(payload.get("judgments"), list):
        return None
    normalized: Dict[str, Dict[str, str]] = {}
    for item in payload["judgments"]:
        if not isinstance(item, dict):
            return None
        unit_id, label = item.get("id"), item.get("label")
        if unit_id not in expected_ids or label not in VALID_LABELS or unit_id in normalized:
            return None
        normalized[unit_id] = {"label": label, "reason": str(item.get("reason", ""))}
    return normalized if set(normalized) == expected_ids else None


def request_audit(endpoint: str, api_key: str, model: str, language: str, units: List[Dict[str, Any]], max_retries: int, temperature: float) -> Optional[Dict[str, Dict[str, str]]]:
    """Submit one audit request.

    Each worker creates its own client to avoid assuming that a third-party
    OpenAI-compatible client is thread-safe. Dataset writes occur only in the
    main thread after a completed future is collected.
    """
    client = OpenAI(base_url=endpoint, api_key=api_key)
    prompt = audit_prompt(TARGET_LANGUAGES[language], units)
    expected_ids = {unit["id"] for unit in units}
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            result = normalize_response(parse_model_json(response.choices[0].message.content), expected_ids)
            if result is not None:
                return result
            tqdm.write(f"Invalid audit response ({attempt}/{max_retries}) for {language}.")
        except Exception as error:
            tqdm.write(f"API error ({attempt}/{max_retries}) for {language}: {error}")
        if attempt < max_retries:
            time.sleep(2 ** attempt)
    return None


def save_judgments(case: Dict[str, Any], language: str, units: List[Dict[str, Any]], judgments: Dict[str, Dict[str, str]], model: str) -> None:
    suffix = f"_{language}"
    question_key = f"questions{suffix}_translation_judgements"
    question_judgments = case.get(question_key, [])
    if not isinstance(question_judgments, list):
        question_judgments = []
    while len(question_judgments) < len(case[f"questions{suffix}"]):
        question_judgments.append({})

    for unit in units:
        judgment = judgments[unit["id"]]
        if unit["id"].startswith("multi_hop_question_"):
            index = int(unit["id"].rsplit("_", 1)[1])
            question_judgments[index] = judgment
            continue
        unit["container"][validation_key(unit)] = judgment["label"]
        unit["container"][reason_key(unit)] = judgment["reason"]
    case[question_key] = question_judgments

    case[f"translation_validation_{language}"] = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_model": model,
        "scope": "automated_llm_audit_not_bilingual_human_validation",
        "labels": sorted(VALID_LABELS),
    }


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    with args.dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("The dataset must be a JSON array.")

    start, stop = args.case_id
    cases = [case for case in data if start <= case.get("case_id", -1) <= stop]
    jobs = []
    for case in cases:
        for language in args.languages:
            try:
                units = build_units(case, language)
            except (KeyError, TypeError, IndexError) as error:
                tqdm.write(f"Skipping case_id={case.get('case_id')} language={language}: {error}")
                continue
            units_to_audit = units if args.overwrite else incomplete_units(units)
            if units_to_audit:
                jobs.append((case, language, units_to_audit))

    skipped = len(cases) * len(args.languages) - len(jobs)
    if skipped:
        print(f"Skipping {skipped} completed or structurally incomplete case-language pair(s).")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                request_audit,
                args.endpoint,
                args.api_key,
                args.model,
                language,
                units_to_audit,
                args.max_retries,
                args.temperature,
            ): (case, language, units_to_audit)
            for case, language, units_to_audit in jobs
        }
        with tqdm(total=len(futures), desc="Auditing translations", unit="case-language") as progress:
            for future in as_completed(futures):
                case, language, units_to_audit = futures[future]
                try:
                    judgments = future.result()
                except Exception as error:
                    tqdm.write(f"Unexpected worker failure for case_id={case['case_id']} "
                               f"language={language}: {error}")
                    judgments = None
                if judgments is None:
                    tqdm.write(f"No judgments saved for case_id={case['case_id']} language={language}; rerun to retry.")
                else:
                    save_judgments(case, language, units_to_audit, judgments, args.model)
                    atomic_write_json(args.dataset_path, data)
                progress.update(1)


if __name__ == "__main__":
    main()
