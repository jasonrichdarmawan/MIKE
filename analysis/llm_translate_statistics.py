"""Summarize embedded translation-audit judgments and calculate confidence intervals.

The validator stores per-unit judgments in the translated dataset. This script
reads those fields without modifying the dataset and writes a compact JSON and
Markdown report. The default analysis unit is the individual translated text
unit; use the case-level summary as a conservative robustness check because
units in the same case are not independent.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LANGUAGES = ("zh", "de", "fr", "ja")
LABELS = ("accepted", "minor_issue", "major_issue")
Z_95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate translation-audit counts and Wilson confidence intervals.")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Defaults to <dataset>.translation_validation_statistics.json.",
    )
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    parser.add_argument(
        "--case-id", nargs=2, type=int, metavar=("START", "STOP"), default=None,
        help="Optional inclusive case_id range, useful for analysing a planned audit sample.",
    )
    return parser.parse_args()


def wilson_interval(successes: int, total: int) -> Optional[Tuple[float, float]]:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + Z_95 ** 2 / total
    centre = (p + Z_95 ** 2 / (2 * total)) / denominator
    half_width = Z_95 * math.sqrt(p * (1 - p) / total + Z_95 ** 2 / (4 * total ** 2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def rate_summary(counts: Counter) -> Dict[str, Any]:
    total = sum(counts.values())
    report: Dict[str, Any] = {"sample_size": total}
    for label in LABELS:
        count = counts[label]
        interval = wilson_interval(count, total)
        report[label] = {
            "count": count,
            "proportion": count / total if total else None,
            "ci_95_wilson": list(interval) if interval else None,
        }
    return report


def record(records: List[Tuple[str, str]], language: str, prompt_type: str, label: Any) -> None:
    if label in LABELS:
        records.append((language, prompt_type, label))


def collect_records(case: Dict[str, Any], language: str) -> List[Tuple[str, str, str]]:
    """Read every stored validation label from one case-language pair."""
    suffix = f"_{language}"
    records: List[Tuple[str, str, str]] = []

    for judgment in case.get(f"questions{suffix}_translation_judgements", []):
        if isinstance(judgment, dict):
            record(records, language, "multi_hop_question", judgment.get("label"))
    record(records, language, "multi_hop_cloze", case.get(f"cloze_question{suffix}_translation_judgement"))

    for field_name in ("answer", "new_answer"):
        record(records, language, "answer", case.get(f"{field_name}{suffix}_translation_judgement"))
    for field_name in ("answer_alias", "new_answer_alias"):
        record(records, language, "answer_alias", case.get(f"{field_name}{suffix}_translation_judgement"))

    for hop_group in (f"single_hops{suffix}", f"new_single_hops{suffix}"):
        for hop in case.get(hop_group, []):
            if not isinstance(hop, dict):
                continue
            record(records, language, "single_hop_question", hop.get("question_translation_judgement"))
            record(records, language, "single_hop_cloze", hop.get("cloze_translation_judgement"))
            record(records, language, "answer", hop.get("answer_translation_judgement"))
            record(records, language, "answer_alias", hop.get("answer_alias_translation_judgement"))
    return records


def case_level_label(records: Iterable[Tuple[str, str, str]]) -> Optional[str]:
    labels = [label for _, _, label in records]
    if not labels:
        return None
    if "major_issue" in labels:
        return "major_issue"
    if "minor_issue" in labels:
        return "minor_issue"
    return "accepted"


def markdown_table(title: str, summary: Dict[str, Any]) -> List[str]:
    lines = [f"### {title}", "", "| Outcome | Count | Proportion | 95% Wilson CI |", "|---|---:|---:|---:|"]
    for label in LABELS:
        item = summary[label]
        interval = item["ci_95_wilson"]
        ci_text = "—" if interval is None else f"[{interval[0]:.1%}, {interval[1]:.1%}]"
        proportion = "—" if item["proportion"] is None else f"{item['proportion']:.1%}"
        lines.append(f"| {label} | {item['count']} | {proportion} | {ci_text} |")
    lines.append(f"| **Total audited units** | **{summary['sample_size']}** | — | — |")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    with args.dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("The dataset must be a JSON array.")
    if args.case_id:
        start, stop = args.case_id
        data = [case for case in data if start <= case.get("case_id", -1) <= stop]

    grouped: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    overall_by_type: Dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    cases_audited = Counter()
    case_level = defaultdict(Counter)

    for case in data:
        for language in args.languages:
            records = collect_records(case, language)
            if records:
                cases_audited[language] += 1
                label = case_level_label(records)
                if label:
                    case_level[language][label] += 1
            for _, prompt_type, label in records:
                grouped[(language, prompt_type)][label] += 1
                overall_by_type[prompt_type][label] += 1
                overall[label] += 1

    prompt_types = (
        "multi_hop_question", "multi_hop_cloze", "single_hop_question",
        "single_hop_cloze", "answer", "answer_alias",
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Automated LLM audit only; this report is not bilingual human validation. "
            "Wilson intervals quantify sampling uncertainty conditional on the audited units."
        ),
        "dataset_path": str(args.dataset_path.resolve()),
        "case_id_range": args.case_id,
        "confidence_interval": "two-sided 95% Wilson binomial interval",
        "by_language_and_prompt_type": {
            language: {prompt_type: rate_summary(grouped[(language, prompt_type)]) for prompt_type in prompt_types}
            for language in args.languages
        },
        "overall_by_prompt_type": {prompt_type: rate_summary(overall_by_type[prompt_type]) for prompt_type in prompt_types},
        "overall_unit_level": rate_summary(overall),
        "case_level_any_issue": {
            language: rate_summary(case_level[language]) for language in args.languages
        },
        "audited_case_language_pairs": dict(cases_audited),
    }

    output_path = args.output_path or args.dataset_path.with_name(args.dataset_path.stem + ".translation_validation_statistics.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Translation Validation Statistics",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "**Scope:** Automated LLM audit only; not bilingual human validation. The 95% Wilson intervals quantify sampling uncertainty conditional on the audited units.",
        "",
    ]
    lines += markdown_table("Overall unit-level results", report["overall_unit_level"])
    for language in args.languages:
        lines += markdown_table(f"{language}: case-level any-issue result", report["case_level_any_issue"][language])
        for prompt_type in prompt_types:
            lines += markdown_table(f"{language}: {prompt_type}", report["by_language_and_prompt_type"][language][prompt_type])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report["overall_unit_level"], ensure_ascii=False, indent=2))
    print(f"Saved JSON statistics: {output_path}")
    print(f"Saved Markdown statistics: {markdown_path}")


if __name__ == "__main__":
    main()
