"""Compare multiple translation-audit judges at unit and language levels.

The input dataset must use the multi-judge schema created by
``migrate_translation_judgements.py`` and ``llm_translate_validate.py``.
Reports include pairwise exact agreement, three-class Cohen's kappa, weighted
kappa, label marginals, a full confusion matrix, and directional label shifts.
"""

# %%

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm_translate_validate import TARGET_LANGUAGES, VALID_LABELS, build_units, judgments_key

# %%

LABELS = ("accepted", "minor_issue", "major_issue")
LABEL_RANK = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate agreement between translation-audit judges.")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--judge-b", required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--languages", nargs="+", choices=tuple(TARGET_LANGUAGES), default=tuple(TARGET_LANGUAGES))
    parser.add_argument("--case-id", nargs=2, type=int, metavar=("START", "STOP"), default=None)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help=(
            "Optional case-clustered bootstrap resamples for exact-agreement "
            "CIs (default: 0, disabled). A CI is usually unnecessary when the "
            "audited units are treated as the complete benchmark population."
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    return parser.parse_args()

# %%

def unit_judgments(unit: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    if unit["id"].startswith("multi_hop_question_"):
        entries = unit["container"].get(unit["key"], [])
        index = int(unit["id"].rsplit("_", 1)[1])
        value = entries[index] if isinstance(entries, list) and len(entries) > index else {}
    else:
        value = unit["container"].get(judgments_key(unit), {})
    return value if isinstance(value, dict) else {}


def collect_pairs(data: Iterable[Dict[str, Any]], languages: Iterable[str], judge_a: str, judge_b: str) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for case in data:
        for language in languages:
            try:
                units = build_units(case, language)
            except (KeyError, TypeError, IndexError):
                continue
            for unit in units:
                judge_map = unit_judgments(unit)
                a = judge_map.get(judge_a, {})
                b = judge_map.get(judge_b, {})
                label_a = a.get("label") if isinstance(a, dict) else None
                label_b = b.get("label") if isinstance(b, dict) else None
                if label_a not in VALID_LABELS or label_b not in VALID_LABELS:
                    continue
                pairs.append({
                    "case_id": case.get("case_id"),
                    "language": language,
                    "unit_id": unit["id"],
                    "prompt_type": unit["prompt_type"],
                    "label_a": label_a,
                    "label_b": label_b,
                })
    return pairs


def cohen_kappa(confusion: Dict[str, Dict[str, int]]) -> Optional[float]:
    total = sum(confusion[a][b] for a in LABELS for b in LABELS)
    if not total:
        return None
    observed = sum(confusion[label][label] for label in LABELS) / total
    row = {label: sum(confusion[label][b] for b in LABELS) for label in LABELS}
    col = {label: sum(confusion[a][label] for a in LABELS) for label in LABELS}
    expected = sum(row[label] * col[label] for label in LABELS) / (total * total)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def weighted_kappa(confusion: Dict[str, Dict[str, int]]) -> Optional[float]:
    """Quadratic-weighted Cohen's kappa for ordered severity labels."""
    total = sum(confusion[a][b] for a in LABELS for b in LABELS)
    if not total:
        return None
    row = {label: sum(confusion[label][b] for b in LABELS) for label in LABELS}
    col = {label: sum(confusion[a][label] for a in LABELS) for label in LABELS}
    denominator_scale = (len(LABELS) - 1) ** 2
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for a in LABELS:
        for b in LABELS:
            weight = (LABEL_RANK[a] - LABEL_RANK[b]) ** 2 / denominator_scale
            observed_disagreement += weight * confusion[a][b] / total
            expected_disagreement += weight * row[a] * col[b] / (total * total)
    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None
    return 1 - observed_disagreement / expected_disagreement


def percentile(sorted_values: List[float], probability: float) -> float:
    """Linearly interpolated percentile for an already sorted sample."""
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from an empty sample.")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def clustered_bootstrap_exact_agreement(
    pairs: List[Dict[str, Any]],
    samples: int,
    seed: int,
) -> Optional[List[float]]:
    """Resample cases and retain all paired units belonging to sampled cases.

    Text units within a benchmark case are correlated. Resampling individual
    units would treat them as independent and produce intervals that are too
    narrow, so the case is the bootstrap cluster.
    """
    if not pairs or samples <= 0:
        return None
    by_case: Dict[Any, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for pair in pairs:
        matched, total = by_case[pair["case_id"]]
        by_case[pair["case_id"]] = (
            matched + int(pair["label_a"] == pair["label_b"]),
            total + 1,
        )
    clusters = list(by_case.values())
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(samples):
        matched = 0
        total = 0
        for _ in range(len(clusters)):
            cluster_matched, cluster_total = clusters[rng.randrange(len(clusters))]
            matched += cluster_matched
            total += cluster_total
        if total:
            estimates.append(matched / total)
    estimates.sort()
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)] if estimates else None


def summarize(
    pairs: List[Dict[str, Any]],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260922,
) -> Dict[str, Any]:
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    marginals_a = Counter()
    marginals_b = Counter()
    exact = 0
    one_step = 0
    two_step = 0
    b_more_severe = 0
    a_more_severe = 0
    for pair in pairs:
        a, b = pair["label_a"], pair["label_b"]
        confusion[a][b] += 1
        marginals_a[a] += 1
        marginals_b[b] += 1
        distance = abs(LABEL_RANK[a] - LABEL_RANK[b])
        exact += distance == 0
        one_step += distance == 1
        two_step += distance == 2
        b_more_severe += LABEL_RANK[b] > LABEL_RANK[a]
        a_more_severe += LABEL_RANK[a] > LABEL_RANK[b]
    total = len(pairs)
    return {
        "paired_units": total,
        "exact_agreement_count": exact,
        "exact_agreement_rate": exact / total if total else None,
        "exact_agreement_ci_95_case_clustered_bootstrap": clustered_bootstrap_exact_agreement(
            pairs,
            bootstrap_samples,
            bootstrap_seed,
        ),
        "cohen_kappa_three_class": cohen_kappa(confusion),
        "quadratic_weighted_kappa": weighted_kappa(confusion),
        "severity_distance": {
            "same": exact,
            "one_level": one_step,
            "two_levels": two_step,
        },
        "directional_disagreement": {
            "judge_a_more_severe": a_more_severe,
            "judge_b_more_severe": b_more_severe,
        },
        "label_counts": {
            "judge_a": {label: marginals_a[label] for label in LABELS},
            "judge_b": {label: marginals_b[label] for label in LABELS},
            "difference_b_minus_a": {label: marginals_b[label] - marginals_a[label] for label in LABELS},
        },
        "confusion_matrix_rows_judge_a_columns_judge_b": confusion,
    }


def markdown_summary(title: str, summary: Dict[str, Any]) -> List[str]:
    rate = summary["exact_agreement_rate"]
    interval = summary["exact_agreement_ci_95_case_clustered_bootstrap"]
    interval_text = (
        f" [95% case-clustered bootstrap CI: {interval[0]:.1%}, {interval[1]:.1%}]"
        if interval is not None
        else ""
    )
    lines = [
        f"## {title}",
        "",
        f"Paired units: **{summary['paired_units']:,}**  ",
        f"Exact agreement: **{rate:.1%}**{interval_text}  " if rate is not None else "Exact agreement: —  ",
        f"Cohen's κ (three classes): **{summary['cohen_kappa_three_class']:.3f}**  " if summary["cohen_kappa_three_class"] is not None else "Cohen's κ: —  ",
        f"Quadratic-weighted κ: **{summary['quadratic_weighted_kappa']:.3f}**" if summary["quadratic_weighted_kappa"] is not None else "Quadratic-weighted κ: —",
        "",
        "| Judge A \\ Judge B | accepted | minor_issue | major_issue |",
        "|---|---:|---:|---:|",
    ]
    matrix = summary["confusion_matrix_rows_judge_a_columns_judge_b"]
    for label in LABELS:
        lines.append(f"| {label} | {matrix[label]['accepted']} | {matrix[label]['minor_issue']} | {matrix[label]['major_issue']} |")
    lines += ["", "| Label | Judge A | Judge B | B − A |", "|---|---:|---:|---:|"]
    counts = summary["label_counts"]
    for label in LABELS:
        lines.append(f"| {label} | {counts['judge_a'][label]} | {counts['judge_b'][label]} | {counts['difference_b_minus_a'][label]:+d} |")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    if args.judge_a == args.judge_b:
        raise ValueError("--judge-a and --judge-b must differ.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative.")
    with args.dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("The dataset must be a JSON array.")
    if args.case_id:
        start, stop = args.case_id
        data = [case for case in data if start <= case.get("case_id", -1) <= stop]

    pairs = collect_pairs(data, args.languages, args.judge_a, args.judge_b)
    by_language = {
        language: summarize(
            [pair for pair in pairs if pair["language"] == language],
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        for language in args.languages
    }
    by_prompt_type = {
        prompt_type: summarize(
            [pair for pair in pairs if pair["prompt_type"] == prompt_type],
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        for prompt_type in sorted({pair["prompt_type"] for pair in pairs})
    }
    prompt_types = sorted({pair["prompt_type"] for pair in pairs})
    by_language_and_prompt_type = {
        language: {
            prompt_type: summarize(
                [
                    pair
                    for pair in pairs
                    if pair["language"] == language and pair["prompt_type"] == prompt_type
                ],
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            for prompt_type in prompt_types
        }
        for language in args.languages
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Agreement between two automated LLM translation-audit judges; not bilingual human validation.",
        "dataset_path": str(args.dataset_path.resolve()),
        "judge_a": args.judge_a,
        "judge_b": args.judge_b,
        "labels_in_severity_order": list(LABELS),
        "case_id_range": args.case_id,
        "exact_agreement_confidence_interval": {
            "method": "two-sided 95% percentile bootstrap clustered by case_id",
            "resamples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interpretation": (
                "Optional superpopulation uncertainty across benchmark cases; "
                "not needed for the observed agreement on the fully audited benchmark."
            ),
        },
        "overall_unit_level": summarize(pairs, args.bootstrap_samples, args.bootstrap_seed),
        "by_language": by_language,
        "by_prompt_type": by_prompt_type,
        "by_language_and_prompt_type": by_language_and_prompt_type,
    }

    output_path = args.output_path or args.dataset_path.with_name(args.dataset_path.stem + ".translation_judge_agreement.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Translation-Judge Agreement",
        "",
        f"Judge A: `{args.judge_a}`  ",
        f"Judge B: `{args.judge_b}`",
        "",
        "**Scope:** Agreement between automated LLM judges; this is not bilingual human validation.",
        "",
    ]
    lines += markdown_summary("Overall unit level", report["overall_unit_level"])
    for language in args.languages:
        lines += markdown_summary(f"Language: {language}", by_language[language])
        for prompt_type in prompt_types:
            lines += markdown_summary(
                f"Language: {language}; unit type: {prompt_type}",
                by_language_and_prompt_type[language][prompt_type],
            )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report["overall_unit_level"], ensure_ascii=False, indent=2))
    print(f"Saved JSON agreement report: {output_path}")
    print(f"Saved Markdown agreement report: {markdown_path}")


if __name__ == "__main__":
    main()
