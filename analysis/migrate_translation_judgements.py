"""Migrate legacy single-judge translation audits to multi-judge mappings.

The migration is idempotent and writes to a separate output path by default.
Legacy ``*_translation_judgement`` and ``*_translation_judgement_reason`` fields
are moved into ``*_translation_judgements[legacy_judge_id]``. Existing
multi-judge entries are preserved unless ``--overwrite-judge`` is supplied.
"""

# %%

import argparse
import json
from pathlib import Path

from typing import Any, Dict, List
import sys

from llm_translate_validate import TARGET_LANGUAGES, VALID_LABELS, atomic_write_json, build_units, judgments_key

# %%

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate translation audit labels to a multi-judge schema.")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Defaults to <dataset>.multi_judge.json; use the dataset path explicitly to replace it.",
    )
    parser.add_argument(
        "--legacy-judge-id",
        required=True,
        help="Stable ID assigned to the existing single-judge labels, e.g. deepseek-v4-pro.",
    )
    parser.add_argument("--languages", nargs="+", choices=tuple(TARGET_LANGUAGES), default=tuple(TARGET_LANGUAGES))
    parser.add_argument("--overwrite-judge", action="store_true")
    parser.add_argument(
        "--keep-legacy-fields",
        action="store_true",
        help="Keep the old label/reason fields after copying them into the new mapping.",
    )
    return parser.parse_args(argv)

# %%

def legacy_label_key(unit: Dict[str, Any]) -> str:
    if unit["id"].startswith("multi_hop_question_"):
        return "label"
    return f"{unit['key']}_translation_judgement"


def legacy_reason_key(unit: Dict[str, Any]) -> str:
    if unit["id"].startswith("multi_hop_question_"):
        return "reason"
    return f"{unit['key']}_translation_judgement_reason"


def migrate_unit(unit: Dict[str, Any], judge_id: str, overwrite: bool, keep_legacy: bool) -> str:
    """Return `migrated`, `preserved`, `missing`, or `invalid`."""
    container = unit["container"]
    if unit["id"].startswith("multi_hop_question_"):
        entries = container.get(unit["key"], [])
        index = int(unit["id"].rsplit("_", 1)[1])
        if not isinstance(entries, list) or len(entries) <= index or not isinstance(entries[index], dict):
            return "missing"
        legacy_container = entries[index]
    else:
        legacy_container = container

    label_key = legacy_label_key(unit)
    reason_key = legacy_reason_key(unit)
    label = legacy_container.get(label_key)
    if label is None:
        return "missing"
    if label not in VALID_LABELS:
        return "invalid"

    if unit["id"].startswith("multi_hop_question_"):
        judge_map = {
            key: value
            for key, value in legacy_container.items()
            if isinstance(value, dict) and value.get("label") in VALID_LABELS
        }
    else:
        mapping_key = judgments_key(unit)
        judge_map = container.get(mapping_key, {})
    if not isinstance(judge_map, dict):
        judge_map = {}
    if judge_id in judge_map and not overwrite:
        return "preserved"

    judge_map[judge_id] = {
        "label": label,
        "reason": str(legacy_container.get(reason_key, "")),
    }
    if unit["id"].startswith("multi_hop_question_"):
        entries[index] = judge_map
    else:
        container[mapping_key] = judge_map

    if not keep_legacy and not unit["id"].startswith("multi_hop_question_"):
        legacy_container.pop(label_key, None)
        legacy_container.pop(reason_key, None)
    return "migrated"

# %%

def main() -> None:
    ARGV = (
        [
            "--dataset-path", "/Users/jason/Library/Mobile Documents/com~apple~CloudDocs/20260915-MIKE-minor-revision/Source-Code/datasets/MQuAKE-CF-3k-v2-clke.json",
            "--legacy-judge-id", "deepseek-v4-pro",
        ]
        if "ipykernel" in sys.modules
        else None
    )

    args = parse_args(ARGV)
    with args.dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("The dataset must be a JSON array.")

    counts = {"migrated": 0, "preserved": 0, "missing": 0, "invalid": 0, "structural_error": 0}
    for case in data:
        for language in args.languages:
            try:
                units = build_units(case, language)
            except (KeyError, TypeError, IndexError):
                counts["structural_error"] += 1
                continue
            for unit in units:
                outcome = migrate_unit(
                    unit,
                    args.legacy_judge_id,
                    args.overwrite_judge,
                    args.keep_legacy_fields,
                )
                counts[outcome] += 1

            metadata_key = f"translation_validation_{language}"
            metadata = case.get(metadata_key, {})
            if isinstance(metadata, dict) and "auditor_model" in metadata:
                judges = metadata.get("judges", {})
                if not isinstance(judges, dict):
                    judges = {}
                judges.setdefault(args.legacy_judge_id, {
                    "audited_at_utc": metadata.get("audited_at_utc"),
                    "auditor_model": metadata.get("auditor_model"),
                    "scope": metadata.get("scope", "automated_llm_audit_not_bilingual_human_validation"),
                    "labels": metadata.get("labels", sorted(VALID_LABELS)),
                    "prompt_version": "legacy_translation_audit_v1",
                })
                case[metadata_key] = {"judges": judges}

    output_path = args.output_path or args.dataset_path.with_name(args.dataset_path.stem + ".multi_judge.json")
    atomic_write_json(output_path, data)
    print(json.dumps(counts, indent=2))
    print(f"Saved migrated dataset: {output_path}")


if __name__ == "__main__":
    main()

# %%