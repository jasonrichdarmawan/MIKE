"""Analyze MQuAKE question-versus-cloze accuracy from ``metrics.json`` files.

For every case, ``post.hop_wise`` contains two outcomes per single-hop fact:
the first is the cloze prompt and the second is the question prompt.
``post.accuracy`` contains one multi-hop cloze outcome followed by three
paraphrased-question outcomes. This script keeps those prompt outcomes grouped
within their case, calculates case-level means, and uses case-level bootstrap
resampling for method-level uncertainty. When a comparison method is supplied,
the same case IDs are resampled for a paired target-minus-comparison analysis.

The CLI accepts either one language-level ``metrics.json`` or an experiment
root containing ``en``, ``de``, ``fr``, ``ja``, and ``zh`` subdirectories.
It writes JSON and Markdown reports without modifying the source metrics.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LANGUAGES = ("en", "de", "fr", "ja", "zh")
METRIC_NAMES = (
	"single_hop_question_accuracy",
	"single_hop_cloze_accuracy",
	"multi_hop_question_accuracy",
	"multi_hop_cloze_accuracy",
)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 20260730


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute MQuAKE question/cloze accuracy by language."
	)
	source_group = parser.add_mutually_exclusive_group(required=True)
	source_group.add_argument(
		"--experiment-root",
		type=Path,
		help="Experiment directory containing <language>/metrics.json.",
	)
	source_group.add_argument(
		"--metrics-path",
		type=Path,
		help="One language-level metrics.json file.",
	)
	parser.add_argument(
		"--comparison-experiment-root",
		type=Path,
		default=None,
		help="Optional comparison root containing <language>/metrics.json.",
	)
	parser.add_argument(
		"--comparison-metrics-path",
		type=Path,
		default=None,
		help="Optional comparison metrics.json when --metrics-path is used.",
	)
	parser.add_argument(
		"--reference-experiment-root",
		type=Path,
		default=None,
		help=(
			"Optional third method root, such as the unedited model, containing "
			"<language>/metrics.json. It is summarized but not used for the "
			"target-minus-comparison contrast."
		),
	)
	parser.add_argument(
		"--reference-metrics-path",
		type=Path,
		default=None,
		help="Optional third method metrics.json when --metrics-path is used.",
	)
	parser.add_argument("--target-label", default=None)
	parser.add_argument("--comparison-label", default=None)
	parser.add_argument("--reference-label", default=None)
	parser.add_argument(
		"--languages",
		nargs="+",
		choices=LANGUAGES,
		default=LANGUAGES,
	)
	parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=None,
		help=(
			"JSON output path. For --experiment-root, defaults to "
			"<root>/question_cloze_statistics.json; for --metrics-path, "
			"defaults beside the metrics file."
		),
	)
	return parser.parse_args()


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	with temporary.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)
	temporary.replace(path)


def require_boolean_list(value: Any, field: str, case_id: Any) -> List[bool]:
	if not isinstance(value, list) or not value:
		raise ValueError(f"case_id={case_id} has empty/non-list {field}.")
	if any(not isinstance(item, bool) for item in value):
		raise ValueError(f"case_id={case_id} has non-boolean values in {field}.")
	return value


def case_prompt_type_means(case: Mapping[str, Any]) -> Dict[str, float]:
	"""Extract four case-level prompt-type accuracies from one MQuAKE case."""
	case_id = case.get("case_id")
	try:
		post = case["post"]
		hop_wise = post["hop_wise"]
		accuracy = post["accuracy"]
	except KeyError as error:
		raise ValueError(f"case_id={case_id} is missing {error}.") from error

	if not isinstance(hop_wise, list) or not hop_wise:
		raise ValueError(f"case_id={case_id} has empty/non-list post.hop_wise.")
	single_hop_cloze: List[bool] = []
	single_hop_question: List[bool] = []
	for hop_index, hop in enumerate(hop_wise):
		values = require_boolean_list(hop, f"post.hop_wise[{hop_index}]", case_id)
		if len(values) != 2:
			raise ValueError(
				f"case_id={case_id} post.hop_wise[{hop_index}] must contain "
				f"[cloze, question], found {len(values)} outcomes."
			)
		single_hop_cloze.append(values[0])
		single_hop_question.append(values[1])

	multi_hop = require_boolean_list(accuracy, "post.accuracy", case_id)
	if len(multi_hop) != 4:
		raise ValueError(
			f"case_id={case_id} post.accuracy must contain one cloze followed "
			f"by three questions, found {len(multi_hop)} outcomes."
		)

	return {
		"single_hop_question_accuracy": float(np.mean(single_hop_question)),
		"single_hop_cloze_accuracy": float(np.mean(single_hop_cloze)),
		"multi_hop_question_accuracy": float(np.mean(multi_hop[1:])),
		"multi_hop_cloze_accuracy": float(multi_hop[0]),
	}


def extract_case_metrics(data: Any) -> Tuple[List[int], Dict[str, np.ndarray]]:
	"""Return ordered case IDs and one case-level array per prompt type."""
	if not isinstance(data, list) or not data:
		raise ValueError("metrics.json must be a non-empty JSON array.")
	seen = set()
	case_ids: List[int] = []
	values: Dict[str, List[float]] = {metric: [] for metric in METRIC_NAMES}
	for case in data:
		if not isinstance(case, dict):
			raise ValueError("Every metrics.json item must be a JSON object.")
		case_id = case.get("case_id")
		if not isinstance(case_id, int):
			raise ValueError(f"Invalid case_id: {case_id!r}")
		if case_id in seen:
			raise ValueError(f"Duplicate case_id in metrics.json: {case_id}")
		seen.add(case_id)
		case_values = case_prompt_type_means(case)
		case_ids.append(case_id)
		for metric in METRIC_NAMES:
			values[metric].append(case_values[metric])
	return case_ids, {metric: np.asarray(items, dtype=float) for metric, items in values.items()}


def percentile_interval(values: np.ndarray) -> List[float]:
	return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def bootstrap_means(
	values: np.ndarray,
	bootstrap_samples: int,
	rng: np.random.Generator,
) -> np.ndarray:
	case_count = len(values)
	means = np.empty(bootstrap_samples, dtype=float)
	chunk_size = max(1, min(1_000, bootstrap_samples))
	for start in range(0, bootstrap_samples, chunk_size):
		stop = min(start + chunk_size, bootstrap_samples)
		indices = rng.integers(0, case_count, size=(stop - start, case_count))
		means[start:stop] = values[indices].mean(axis=1)
	return means


def metric_summary(
	values: np.ndarray,
	bootstrap_samples: int,
	rng: np.random.Generator,
) -> Dict[str, Any]:
	estimate = float(values.mean())
	interval = percentile_interval(bootstrap_means(values, bootstrap_samples, rng))
	return {
		"estimate": estimate,
		"estimate_percent": estimate * 100,
		"ci_95_case_bootstrap": interval,
		"ci_95_case_bootstrap_percent": [value * 100 for value in interval],
		"case_count": int(len(values)),
	}


def align_comparison(
	comparison_data: Any,
	target_case_ids: Sequence[int],
) -> Dict[str, np.ndarray]:
	comparison_ids, comparison_metrics = extract_case_metrics(comparison_data)
	target_id_set = set(target_case_ids)
	if set(comparison_ids) != target_id_set:
		missing = sorted(target_id_set - set(comparison_ids))
		extra = sorted(set(comparison_ids) - target_id_set)
		raise ValueError(
			"Paired bootstrap requires identical case IDs. "
			f"Missing from comparison: {missing[:10]}; extra: {extra[:10]}."
		)
	position = {case_id: index for index, case_id in enumerate(comparison_ids)}
	return {
		metric: np.asarray(
			[comparison_metrics[metric][position[case_id]] for case_id in target_case_ids],
			dtype=float,
		)
		for metric in METRIC_NAMES
	}


def paired_difference_summary(
	target: np.ndarray,
	comparison: np.ndarray,
	bootstrap_samples: int,
	rng: np.random.Generator,
) -> Dict[str, Any]:
	differences = target - comparison
	estimate = float(differences.mean())
	interval = percentile_interval(bootstrap_means(differences, bootstrap_samples, rng))
	return {
		"target_minus_comparison": estimate,
		"target_minus_comparison_percentage_points": estimate * 100,
		"ci_95_paired_case_bootstrap": interval,
		"ci_95_paired_case_bootstrap_percentage_points": [
			value * 100 for value in interval
		],
		"case_count": int(len(target)),
	}


def analyze_language(
	metrics_path: Path,
	comparison_metrics_path: Optional[Path],
	reference_metrics_path: Optional[Path],
	target_label: str,
	comparison_label: Optional[str],
	reference_label: Optional[str],
	bootstrap_samples: int,
	seed: int,
) -> Dict[str, Any]:
	case_ids, metrics = extract_case_metrics(load_json(metrics_path))
	rng = np.random.default_rng(seed)
	report: Dict[str, Any] = {
		"metrics_path": str(metrics_path.resolve()),
		"case_count": len(case_ids),
		"methods": {
			target_label: {
				"metrics_path": str(metrics_path.resolve()),
				"metrics": {
					metric: metric_summary(metrics[metric], bootstrap_samples, rng)
					for metric in METRIC_NAMES
				},
			}
		},
	}
	if comparison_metrics_path is not None:
		comparison = align_comparison(load_json(comparison_metrics_path), case_ids)
		resolved_comparison_label = comparison_label or comparison_metrics_path.parent.parent.name
		report["methods"][resolved_comparison_label] = {
			"metrics_path": str(comparison_metrics_path.resolve()),
			"metrics": {
				metric: metric_summary(comparison[metric], bootstrap_samples, rng)
				for metric in METRIC_NAMES
			},
		}
		report["paired_differences"] = {
			metric: paired_difference_summary(
				metrics[metric], comparison[metric], bootstrap_samples, rng
			)
			for metric in METRIC_NAMES
		}
		report["paired_difference_contrast"] = (
			f"{target_label} minus {resolved_comparison_label}"
		)
	if reference_metrics_path is not None:
		reference = align_comparison(load_json(reference_metrics_path), case_ids)
		resolved_reference_label = reference_label or reference_metrics_path.parent.parent.name
		report["methods"][resolved_reference_label] = {
			"metrics_path": str(reference_metrics_path.resolve()),
			"metrics": {
				metric: metric_summary(reference[metric], bootstrap_samples, rng)
				for metric in METRIC_NAMES
			},
		}
	return report


def markdown_metric(label: str, summary: Mapping[str, Any]) -> str:
	estimate = summary["estimate_percent"]
	lower, upper = summary["ci_95_case_bootstrap_percent"]
	return f"| {label} | {estimate:.2f}% | [{lower:.2f}%, {upper:.2f}%] |"


def markdown_report(report: Mapping[str, Any]) -> str:
	lines = [
		"# MQuAKE Question-versus-Cloze Accuracy",
		"",
		f"Generated: {report['generated_at_utc']}",
		"",
		(
			"Case-level means are reported with 95% percentile bootstrap "
			"intervals. All prompt outcomes belonging to a sampled case remain together."
		),
		"",
	]
	for language, language_report in report["languages"].items():
		lines += [f"## {language}", ""]
		for method_label, method_report in language_report["methods"].items():
			lines += [
				f"### {method_label}",
				"",
				"| Prompt type | Accuracy | 95% case-bootstrap CI |",
				"|---|---:|---:|",
			]
			for metric in METRIC_NAMES:
				lines.append(markdown_metric(metric, method_report["metrics"][metric]))
			lines.append("")
		if "paired_differences" in language_report:
			lines += [
				f"### {language_report['paired_difference_contrast']}",
				"",
				"| Prompt type | Target − comparison (percentage points) | 95% paired case-bootstrap CI |",
				"|---|---:|---:|",
			]
			for metric in METRIC_NAMES:
				item = language_report["paired_differences"][metric]
				lower, upper = item["ci_95_paired_case_bootstrap_percentage_points"]
				lines.append(
					f"| {metric} | {item['target_minus_comparison_percentage_points']:.2f} | "
					f"[{lower:.2f}, {upper:.2f}] |"
				)
		lines.append("")
	return "\n".join(lines)


def infer_output_path(args: argparse.Namespace) -> Path:
	if args.output_path is not None:
		return args.output_path
	if args.experiment_root is not None:
		return args.experiment_root / "question_cloze_statistics.json"
	return args.metrics_path.with_name("question_cloze_statistics.json")


def main() -> None:
	args = parse_args()
	if args.bootstrap_samples <= 0:
		raise ValueError("--bootstrap-samples must be positive.")
	if args.comparison_experiment_root and not args.experiment_root:
		raise ValueError("--comparison-experiment-root requires --experiment-root.")
	if args.comparison_metrics_path and not args.metrics_path:
		raise ValueError("--comparison-metrics-path requires --metrics-path.")
	if args.comparison_experiment_root and args.comparison_metrics_path:
		raise ValueError("Provide only one comparison path mode.")
	if args.reference_experiment_root and not args.experiment_root:
		raise ValueError("--reference-experiment-root requires --experiment-root.")
	if args.reference_metrics_path and not args.metrics_path:
		raise ValueError("--reference-metrics-path requires --metrics-path.")
	if args.reference_experiment_root and args.reference_metrics_path:
		raise ValueError("Provide only one reference path mode.")

	target_label = args.target_label or (
		args.experiment_root.name if args.experiment_root else args.metrics_path.parent.parent.name
	)

	language_reports: Dict[str, Any] = {}
	if args.experiment_root is not None:
		for language in args.languages:
			metrics_path = args.experiment_root / language / "metrics.json"
			if not metrics_path.exists():
				raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
			comparison_path = None
			if args.comparison_experiment_root is not None:
				comparison_path = args.comparison_experiment_root / language / "metrics.json"
				if not comparison_path.exists():
					raise FileNotFoundError(f"Missing comparison metrics file: {comparison_path}")
			reference_path = None
			if args.reference_experiment_root is not None:
				reference_path = args.reference_experiment_root / language / "metrics.json"
				if not reference_path.exists():
					raise FileNotFoundError(f"Missing reference metrics file: {reference_path}")
			language_reports[language] = analyze_language(
				metrics_path,
				comparison_path,
				reference_path,
				target_label,
				args.comparison_label,
				args.reference_label,
				args.bootstrap_samples,
				args.seed,
			)
	else:
		language_reports[args.metrics_path.parent.name] = analyze_language(
			args.metrics_path,
			args.comparison_metrics_path,
			args.reference_metrics_path,
			target_label,
			args.comparison_label,
			args.reference_label,
			args.bootstrap_samples,
			args.seed,
		)

	report = {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"target_label": target_label,
		"comparison_label": args.comparison_label,
		"bootstrap_samples": args.bootstrap_samples,
		"seed": args.seed,
		"resampling_unit": "case_id with all within-case prompt outcomes retained",
		"prompt_order": {
			"single_hop": "post.hop_wise[*] = [cloze, question]",
			"multi_hop": "post.accuracy = [cloze, question_1, question_2, question_3]",
		},
		"languages": language_reports,
	}
	output_path = infer_output_path(args)
	atomic_write_json(output_path, report)
	markdown_path = output_path.with_suffix(".md")
	markdown_path.write_text(markdown_report(report), encoding="utf-8")
	print(json.dumps(report, ensure_ascii=False, indent=2))
	print(f"Saved JSON statistics: {output_path}")
	print(f"Saved Markdown statistics: {markdown_path}")


if __name__ == "__main__":
	main()
