"""Add case-level bootstrap uncertainty to a knowledge-editing ``res.json``.

The metrics file contains multiple correlated prompt outcomes per edit case.
This script resamples whole cases with replacement, retaining every one-hop and
multi-hop prompt belonging to each sampled case.  It therefore matches the
case-level aggregation used by ``test_cake.calculate_averages`` and avoids
incorrectly treating paraphrases from the same case as independent examples.

For method differences, run this script with ``--comparison-metrics-path``.
The two files must contain the same case IDs; a paired case-level bootstrap CI
for the percentage-point difference is then written to the target ``res.json``.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 20260730


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute case-level bootstrap CIs from a knowledge-editing metrics.json file."
    )
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument(
        "--res-path",
        type=Path,
        default=None,
        help="Output res.json path. Defaults to metrics.json's sibling res.json.",
    )
    parser.add_argument(
        "--do-individual-method",
        action="store_true",
        help="Include case-level bootstrap CIs for the target metrics.json itself.",
    )
    parser.add_argument(
        "--comparison-metrics-path",
        type=Path,
        default=None,
        help=(
            "Optional baseline/other-method metrics.json. Adds paired bootstrap "
            "CIs for this metrics file minus the comparison file."
        ),
    )
    parser.add_argument(
        "--comparison-label",
        default=None,
        help="Human-readable comparison label stored in res.json.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def case_means(case: Dict[str, Any]) -> Dict[str, float]:
    """Reproduce the case-level metrics used to generate the existing res.json."""
    try:
        post = case["post"]
        hop_wise = post["hop_wise"]
        accuracy = post["accuracy"]
    except KeyError as error:
        raise ValueError(f"case_id={case.get('case_id')} is missing {error}.") from error

    hop_values = [value for hop in hop_wise for value in hop]
    if not hop_values or not accuracy:
        raise ValueError(f"case_id={case.get('case_id')} has empty hop_wise or accuracy outcomes.")
    if any(not isinstance(value, bool) for value in hop_values + list(accuracy)):
        raise ValueError(f"case_id={case.get('case_id')} contains missing/non-boolean outcomes.")

    return {
        "single_hop_accuracy": float(np.mean(hop_values)),
        "multi_hop_accuracy": float(np.mean(accuracy)),
    }


def extract_case_metrics(data: Any) -> tuple[np.ndarray, np.ndarray, List[int]]:
    if not isinstance(data, list) or not data:
        raise ValueError("metrics.json must be a non-empty JSON array of evaluation cases.")
    seen_case_ids = set()
    single_hop, multi_hop, case_ids = [], [], []
    for case in data:
        case_id = case.get("case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id in metrics.json: {case_id}")
        seen_case_ids.add(case_id)
        values = case_means(case)
        single_hop.append(values["single_hop_accuracy"])
        multi_hop.append(values["multi_hop_accuracy"])
        case_ids.append(case_id)
    return np.asarray(single_hop), np.asarray(multi_hop), case_ids


def percentile_interval(values: np.ndarray) -> List[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def bootstrap_means(
    metrics: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap means after resampling complete edit cases."""
    case_count = len(metrics)
    # Chunking bounds memory for 10,000 x 1,000 case resamples.
    means = np.empty(bootstrap_samples, dtype=float)
    chunk_size = max(1, min(1_000, bootstrap_samples))
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        sampled_indices = rng.integers(0, case_count, size=(stop - start, case_count))
        means[start:stop] = metrics[sampled_indices].mean(axis=1)
    return means


def metric_summary(
    values: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    bootstrap_values = bootstrap_means(values, bootstrap_samples, rng)
    return {
        "estimate": float(values.mean()),
        "estimate_percent": float(values.mean() * 100),
        "ci_95_percentile_bootstrap": percentile_interval(bootstrap_values),
        "ci_95_percentile_bootstrap_percent": [
            float(value * 100) for value in percentile_interval(bootstrap_values)
        ],
        "case_count": int(len(values)),
    }


def paired_difference_summary(
    target: np.ndarray,
    comparison: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Paired bootstrap for target minus comparison on the same cases."""
    case_count = len(target)
    differences = target - comparison
    bootstrap_differences = bootstrap_means(differences, bootstrap_samples, rng)
    return {
        "target_minus_comparison": float(differences.mean()),
        "target_minus_comparison_percentage_points": float(differences.mean() * 100),
        "ci_95_percentile_paired_bootstrap": percentile_interval(bootstrap_differences),
        "ci_95_percentile_paired_bootstrap_percentage_points": [
            float(value * 100) for value in percentile_interval(bootstrap_differences)
        ],
        "case_count": int(case_count),
    }


def align_comparison(
    comparison_data: Any,
    target_case_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    comparison_single, comparison_multi, comparison_ids = extract_case_metrics(comparison_data)
    comparison_by_id = {
        case_id: (comparison_single[index], comparison_multi[index])
        for index, case_id in enumerate(comparison_ids)
    }
    target_id_set = set(target_case_ids)
    if set(comparison_by_id) != target_id_set:
        missing = sorted(target_id_set - set(comparison_by_id))
        extra = sorted(set(comparison_by_id) - target_id_set)
        raise ValueError(
            "Paired bootstrap requires identical case IDs. "
            f"Missing from comparison: {missing[:10]}; extra in comparison: {extra[:10]}."
        )
    single_hop = np.asarray([comparison_by_id[case_id][0] for case_id in target_case_ids])
    multi_hop = np.asarray([comparison_by_id[case_id][1] for case_id in target_case_ids])
    return single_hop, multi_hop


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")

    target_data = load_json(args.metrics_path)
    single_hop, multi_hop, case_ids = extract_case_metrics(target_data)
    rng = np.random.default_rng(args.seed)

    bootstrap_statistics: Dict[str, Any] = {}
    if args.do_individual_method:
        bootstrap_statistics["individual_method"] = {
            "method": "case-level percentile bootstrap",
            "metrics": {
                "single_hop_accuracy": metric_summary(single_hop, args.bootstrap_samples, rng),
                "multi_hop_accuracy": metric_summary(multi_hop, args.bootstrap_samples, rng),
            },
        }

    if args.comparison_metrics_path:
        comparison_single, comparison_multi = align_comparison(
            load_json(args.comparison_metrics_path), case_ids
        )
        bootstrap_statistics["paired_difference"] = {
            "comparison_metrics_path": str(args.comparison_metrics_path.resolve()),
            "comparison_label": args.comparison_label or args.comparison_metrics_path.parent.name,
            "method": "paired case-level percentile bootstrap",
            "metrics": {
                "single_hop_accuracy": paired_difference_summary(
                    single_hop, comparison_single, args.bootstrap_samples, rng
                ),
                "multi_hop_accuracy": paired_difference_summary(
                    multi_hop, comparison_multi, args.bootstrap_samples, rng
                ),
            },
        }

    res_path = args.res_path or args.metrics_path.with_name("res.json")
    existing_res = load_json(res_path) if res_path.exists() else {}
    if not isinstance(existing_res, dict):
        raise ValueError(f"res.json must be a JSON object: {res_path}")

    existing_res["statistics_uncertainty"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(args.metrics_path.resolve()),
        "confidence_level": 0.95,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "resampling_unit": (
            "evaluation case_id; all one-hop and multi-hop prompt outcomes in a "
            "sampled case are retained together"
        ),
        **bootstrap_statistics,
    }
    atomic_write_json(res_path, existing_res)

    print(json.dumps(existing_res["statistics_uncertainty"], ensure_ascii=False, indent=2))
    print(f"Updated statistics uncertainty in: {res_path}")


if __name__ == "__main__":
    main()
