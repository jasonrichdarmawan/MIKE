"""Generate a figure-independent statistical report for entity-decoding line charts.

The input is ``min_entity_description_layers.csv`` written by
``analysis.analyze_results.plot_min_entity_description_layers``. Each line
contains the percentage of all cases whose *minimum* decoding layer equals a
given layer. Because the plot is constructed with ``dropna=False``, a line's
percentages sum to its detection rate, not necessarily 100%.

Example
-------
python min_entity_description_layers_to_statistics.py /path/to/min_entity_description_layers.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"layer", "percent", "series"}


def weighted_quantile(layers: np.ndarray, weights: np.ndarray, quantile: float) -> float:
	"""Return the first discrete layer whose cumulative mass reaches a quantile."""
	if not weights.sum():
		return np.nan
	return float(layers[np.searchsorted(np.cumsum(weights) / weights.sum(), quantile)])


def shortest_mass_interval(layers: np.ndarray, weights: np.ndarray, mass: float = 0.50) -> tuple[float, float]:
	"""Find the narrowest contiguous layer interval containing at least ``mass``."""
	if not weights.sum():
		return np.nan, np.nan
	probabilities = weights / weights.sum()
	best: tuple[float, float, float] | None = None
	for start in range(len(layers)):
		cumulative = 0.0
		for end in range(start, len(layers)):
			cumulative += probabilities[end]
			if cumulative >= mass:
				candidate = (layers[end] - layers[start], layers[start], layers[end])
				if best is None or candidate < best:
					best = candidate
				break
	return (float(best[1]), float(best[2])) if best else (np.nan, np.nan)


def summarize_series(series: str, group: pd.DataFrame) -> dict[str, object]:
	"""Summarize one line as a distribution over its observed decoding layers."""
	group = group.sort_values("layer")
	layers = group["layer"].to_numpy(dtype=float)
	percent = group["percent"].to_numpy(dtype=float)
	detection_rate = percent.sum()
	if detection_rate <= 0:
		return {"series": series, "layer_count": len(group), "detection_rate_percent": 0.0}

	weights = percent / detection_rate
	peak_percent = percent.max()
	peak_layers = ", ".join(str(int(layer)) for layer in layers[np.isclose(percent, peak_percent)])
	interval_low, interval_high = shortest_mass_interval(layers, percent)
	return {
		"series": series,
		"layer_count": len(group),
		"detection_rate_percent": detection_rate,
		"missing_rate_percent": max(0.0, 100 - detection_rate),
		"conditional_mean_layer": float(np.average(layers, weights=weights)),
		"conditional_median_layer": weighted_quantile(layers, percent, 0.50),
		"conditional_q1_layer": weighted_quantile(layers, percent, 0.25),
		"conditional_q3_layer": weighted_quantile(layers, percent, 0.75),
		"central_50_percent_low_layer": interval_low,
		"central_50_percent_high_layer": interval_high,
		"peak_layer_or_layers": peak_layers,
		"peak_percent_of_all_cases": peak_percent,
		"peak_percent_of_detected_cases": 100 * peak_percent / detection_rate,
		"first_nonzero_layer": float(layers[percent > 0].min()),
		"last_nonzero_layer": float(layers[percent > 0].max()),
	}


def compare_curves(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict[str, object]:
	"""Compare conditional distributions; detection-rate differences are reported separately."""
	merged = left[["layer", "percent"]].merge(
		right[["layer", "percent"]], on="layer", how="outer", suffixes=("_left", "_right")
	).fillna(0).sort_values("layer")
	left_rate, right_rate = merged["percent_left"].sum(), merged["percent_right"].sum()
	result: dict[str, object] = {
		"series_a": left_name,
		"series_b": right_name,
		"detection_rate_difference_a_minus_b_pp": left_rate - right_rate,
	}
	if left_rate <= 0 or right_rate <= 0:
		return result
	left_distribution = merged["percent_left"] / left_rate
	right_distribution = merged["percent_right"] / right_rate
	difference = left_distribution - right_distribution
	max_index = difference.abs().idxmax()
	result.update(
		{
			"total_variation_distance": 0.5 * difference.abs().sum(),
			"largest_conditional_difference_layer": int(merged.loc[max_index, "layer"]),
			"largest_conditional_difference_a_minus_b_pp": 100 * difference.loc[max_index],
		}
	)
	return result


def number(value: object, digits: int = 2) -> str:
	return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def write_report(path: Path, summaries: pd.DataFrame, comparisons: pd.DataFrame) -> None:
	"""Write concise guidance, a worked example, and line-by-line interpretation."""
	lines = [
		"# Minimum entity-description layer line-chart report",
		"",
		"Each point is the percentage of **all** cases whose earliest detected decoding layer equals the x-axis layer. Thus, each line's total height (sum of percentages) is its detection rate; missing/undetected cases are not assigned to any layer.",
		"",
		"## How to read the statistics",
		"",
		"- **Detection / missing rate:** Compute this to separate whether a representation is found from where it is found. A line summing to 60% means the entity was detected in 60% of all cases and undetected in 40%; its shape therefore describes detected cases only.",
		"- **Peak:** The peak layer is the most common earliest decoding layer; `peak_percent_of_all_cases` gives its height in the plotted line, while `peak_percent_of_detected_cases` gives its share after excluding undetected cases.",
		"- **Conditional median:** Compute this to identify the typical earliest layer *when detection occurs*. For a conditional median of layer 10, half of detected cases occur at layer 10 or earlier and half at layer 10 or later; undetected cases are excluded.",
		"- **Narrowest 50% interval:** Compute this to locate the most concentrated contiguous half of detected cases. For an interval of layers 6–13, at least half of detected cases first decode in that range; it can be narrower or shifted relative to Q1–Q3 for skewed distributions.",
		"- **Detection-rate difference (A − B):** Compute this to compare how often two representations are found. For A = 29.58% and B = 17.27%, A − B = +12.31 percentage points means A is detected in 12.31 more cases per 100 total cases.",
		"- **Conditional-shape total variation distance:** Compute this to compare *where* two representations are detected after removing their detection-rate difference. It ranges from 0 (identical detected-case layer distributions) to 1 (no overlap); for example, 0.83 indicates strongly different conditional layer patterns. This is descriptive, not a hypothesis test.",
		"- **Largest conditional difference:** Compute this to show the specific layer driving the overall shape difference. `A − B = −28.60` percentage points at layer 24 means that, among detected cases, A has 28.60 fewer cases per 100 than B at layer 24; the negative sign indicates that layer 24 is more characteristic of B.",
		"",
		"## Detection and representativeness cautions",
		"",
	]
	for _, row in summaries.iterrows():
		if row["missing_rate_percent"] >= 50:
			lines.append(
				f"- **{row['series']}:** detected in only {number(row['detection_rate_percent'], 1)}% of cases "
				f"({number(row['missing_rate_percent'], 1)}% undetected). Its line shape represents detected cases only and should not be generalized to all cases."
			)
	if not any(summaries["missing_rate_percent"] >= 50):
		lines.append("- All series are detected in more than half of cases.")

	lines.extend(["", "## Series summaries", ""])
	for _, row in summaries.iterrows():
		lines.append(
			f"- **{row['series']}:** detection {number(row['detection_rate_percent'])}% "
			f"(undetected {number(row['missing_rate_percent'])}%); peak at layer {row['peak_layer_or_layers']} "
			f"({number(row['peak_percent_of_all_cases'])}% of all; {number(row['peak_percent_of_detected_cases'])}% of detected); "
			f"conditional median {number(row['conditional_median_layer'], 0)} [Q1 {number(row['conditional_q1_layer'], 0)}, Q3 {number(row['conditional_q3_layer'], 0)}]; "
			f"narrowest 50% interval: layers {number(row['central_50_percent_low_layer'], 0)}–{number(row['central_50_percent_high_layer'], 0)}."
		)

	if not comparisons.empty:
		lines.extend(["", "## Pairwise curve comparisons", ""])
		for _, row in comparisons.iterrows():
			if pd.notna(row.get("total_variation_distance")):
				lines.append(
					f"- **{row['series_a']} vs {row['series_b']}:** detection-rate difference "
					f"(A − B) = {number(row['detection_rate_difference_a_minus_b_pp'])} percentage points; "
					f"conditional-shape total variation distance = {number(row['total_variation_distance'])}; largest difference at layer "
					f"{int(row['largest_conditional_difference_layer'])} (A − B = {number(row['largest_conditional_difference_a_minus_b_pp'])} percentage points among detected cases)."
				)

	first = summaries.iloc[0]
	lines.extend([
		"", "## Worked example and paper-ready interpretation", "",
		f"- **Numerical reading — {first['series']}:** The line sums to {number(first['detection_rate_percent'])}%, so the entity is detected in that fraction of all cases. Among detected cases, the median earliest layer is {number(first['conditional_median_layer'], 0)}, and the most frequent earliest layer is {first['peak_layer_or_layers']}.",
		f"> For **{first['series']}**, the entity was detected in {number(first['detection_rate_percent'])}% of cases. Conditional on detection, earliest decoding was concentrated around layer {number(first['conditional_median_layer'], 0)} (central 50% interval: layers {number(first['central_50_percent_low_layer'], 0)}–{number(first['central_50_percent_high_layer'], 0)}). This describes the location of detected representations and does not imply detection in the remaining {number(first['missing_rate_percent'])}% of cases.",
		"",
	])
	path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("csv_path", type=Path, help="Path to min_entity_description_layers.csv")
	parser.add_argument("--output-dir", type=Path, default=None, help="Directory for reports (default: CSV directory)")
	args = parser.parse_args()
	if not args.csv_path.is_file():
		parser.error(f"Input file does not exist: {args.csv_path}")

	data = pd.read_csv(args.csv_path)
	missing_columns = REQUIRED_COLUMNS - set(data.columns)
	if missing_columns:
		parser.error(f"Input is missing required columns: {', '.join(sorted(missing_columns))}")
	data["layer"] = pd.to_numeric(data["layer"], errors="coerce")
	data["percent"] = pd.to_numeric(data["percent"], errors="coerce")
	if data[["layer", "percent"]].isna().any().any() or (data["percent"] < 0).any():
		parser.error("`layer` and `percent` must be non-missing numeric values and percent must be non-negative.")

	summaries = pd.DataFrame(
		[summarize_series(series, group) for series, group in data.groupby("series", sort=False)]
	)
	series_groups = {series: group for series, group in data.groupby("series", sort=False)}
	comparisons = pd.DataFrame(
		[
			compare_curves(series_groups[left], series_groups[right], left, right)
			for index, left in enumerate(series_groups)
			for right in list(series_groups)[index + 1 :]
		]
	)

	output_dir = args.output_dir or args.csv_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)
	summary_path = output_dir / "min_entity_description_layers_statistics.csv"
	comparison_path = output_dir / "min_entity_description_layers_curve_comparisons.csv"
	report_path = output_dir / "min_entity_description_layers_statistical_report.md"
	summaries.to_csv(summary_path, index=False, float_format="%.6f")
	comparisons.to_csv(comparison_path, index=False, float_format="%.6f")
	write_report(report_path, summaries, comparisons)
	print(f"Wrote line statistics: {summary_path}")
	print(f"Wrote curve comparisons: {comparison_path}")
	print(f"Wrote narrative report: {report_path}")


if __name__ == "__main__":
	main()
