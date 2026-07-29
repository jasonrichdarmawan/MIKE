"""Generate a text and tabular statistical description of a stage boxplot.

The input is the ``stages_boxplot.csv`` written by
``analysis.analyze_results.plot_stage_boxplots``.  Its statistics use the same
Tukey convention as seaborn's default boxplot: the box spans Q1--Q3 and
whiskers extend to the most extreme observations within 1.5 IQR of the box.

Example
-------
python stages_boxplot_to_statistics.py /path/to/plots/stages_boxplot.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


REQUIRED_COLUMNS = {"stage", "value", "Two-Hop Query Answer"}
OUTCOME_COLUMN = "Two-Hop Query Answer"


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
	"""Return Benjamini--Hochberg FDR-adjusted p-values, retaining the index."""
	adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
	valid = p_values.dropna()
	if valid.empty:
		return adjusted

	ordered = valid.sort_values()
	ranks = np.arange(1, len(ordered) + 1)
	values = ordered.to_numpy() * len(ordered) / ranks
	# Enforce monotonicity when moving from large to small p-values.
	values = np.minimum.accumulate(values[::-1])[::-1]
	adjusted.loc[ordered.index] = np.minimum(values, 1.0)
	return adjusted


def boxplot_summary(values: pd.Series) -> dict[str, float | int]:
	"""Compute descriptive and exact default-boxplot statistics for one group."""
	observed = values.dropna().astype(float)
	total = len(values)
	missing = total - len(observed)
	if observed.empty:
		return {
			"n_total": total,
			"n_observed": 0,
			"n_missing": missing,
			"missing_percent": 100.0 if total else np.nan,
		}

	q1, median, q3 = observed.quantile([0.25, 0.50, 0.75]).to_numpy()
	iqr = q3 - q1
	lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
	in_fence = observed[(observed >= lower_fence) & (observed <= upper_fence)]
	return {
		"n_total": total,
		"n_observed": len(observed),
		"n_missing": missing,
		"missing_percent": 100 * missing / total if total else np.nan,
		"mean": observed.mean(),
		"std": observed.std(ddof=1) if len(observed) > 1 else np.nan,
		"min": observed.min(),
		"q1": q1,
		"median": median,
		"q3": q3,
		"max": observed.max(),
		"iqr": iqr,
		"lower_fence": lower_fence,
		"upper_fence": upper_fence,
		"lower_whisker": in_fence.min(),
		"upper_whisker": in_fence.max(),
		"outlier_count": int(((observed < lower_fence) | (observed > upper_fence)).sum()),
	}


def compare_outcomes(stage: str, group: pd.DataFrame) -> dict[str, float | int | str]:
	"""Compare Correct with Incorrect using a two-sided Mann--Whitney U test."""
	correct = group.loc[group[OUTCOME_COLUMN] == "Correct", "value"].dropna()
	incorrect = group.loc[group[OUTCOME_COLUMN] == "Incorrect", "value"].dropna()
	result: dict[str, float | int | str] = {
		"stage": stage,
		"correct_n": len(correct),
		"incorrect_n": len(incorrect),
		"median_difference_correct_minus_incorrect": (
			correct.median() - incorrect.median() if len(correct) and len(incorrect) else np.nan
		),
	}
	if not len(correct) or not len(incorrect):
		return result

	test = mannwhitneyu(correct, incorrect, alternative="two-sided", method="auto")
	total_pairs = len(correct) * len(incorrect)
	probability_superiority = test.statistic / total_pairs
	result.update(
		{
			"mann_whitney_u": test.statistic,
			"p_value": test.pvalue,
			# Positive values mean a randomly chosen Correct value tends to be higher.
			"rank_biserial_correct_minus_incorrect": 2 * probability_superiority - 1,
			"probability_correct_greater_than_incorrect": probability_superiority,
		}
	)
	return result


def format_number(value: object, digits: int = 2) -> str:
	return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def write_markdown_report(
	path: Path, summaries: pd.DataFrame, comparisons: pd.DataFrame, alpha: float
) -> None:
	"""Write a concise, figure-independent narrative for the plotted data."""
	lines = [
		"# Stage boxplot statistical report",
		"",
		"This report describes the raw values used in the boxplot. Missing `value`s are "
		"not plotted and are reported explicitly below. In this dataset, a missing value means "
		"that the analysis did not identify the relevant stage/layer for that example; it is not a layer value of zero.",
		"",
		"## How to read the statistics",
		"",
		"- **Box / median / IQR:** The box contains the middle 50% of observed layers (Q1–Q3); its centre is the median and IQR = Q3 − Q1. For example, Q1 = 27, median = 28, Q3 = 29 means half of detected cases lie between layers 27 and 29.",
		"- **Whiskers / outliers:** Whiskers show the most extreme non-outlying observations under Tukey's $1.5\\times$IQR rule; observations beyond them are counted as outliers, which are unusual detected layers rather than errors.",
		"- **Missingness:** `observed n/total n` is the stage-detection rate; it is reported because the boxplot excludes undetected cases. High missingness means the box describes only a potentially selective subset, not all cases in that outcome group.",
		"- **Correct vs Incorrect:** This comparison tests whether detected layers tend to differ by answer outcome, which can descriptively reveal different internal trajectories but cannot establish causality. $U$ and $p$ come from a two-sided Mann--Whitney U test: all observed layers are jointly ranked, then the test asks whether either outcome tends to receive higher ranks; it does not assume normal distributions.",
		"- **Significance:** Use $p_{FDR-BH}$, not the raw $p$, because six stages are tested. If $p_{FDR-BH} < 0.05$, the difference is statistically significant at $\\alpha=0.05$; otherwise describe it as descriptive, not confirmatory.",
		"- **Rank-biserial:** This effect size ranges from −1 to +1; positive means Correct tends to occur at higher layers, negative means Incorrect tends to occur at higher layers, and values near zero indicate little rank separation. Its size describes the tendency, while $p_{FDR-BH}$ describes the evidence against no difference.",
		"",
		"## Detection and representativeness cautions",
		"",
	]

	for _, row in summaries.iterrows():
		if row["missing_percent"] >= 50:
			lines.append(
				f"- **{row['stage']} — {row[OUTCOME_COLUMN]}:** detected for only "
				f"{int(row['n_observed'])}/{int(row['n_total'])} cases "
				f"({format_number(row['missing_percent'], 1)}% missing). Its boxplot distribution "
				"therefore represents only detected cases and should not be interpreted as representative "
				"of all cases in this outcome group."
			)
	if not any(summaries["missing_percent"] >= 50):
		lines.append("- No stage-outcome group has at least 50% missing values.")

	lines.extend(["", "## Worked example and paper-ready interpretation", ""])
	# Use the smallest raw p-value to illustrate the strongest observed contrast without
	# presenting it as a confirmatory finding after multiplicity correction.
	example = comparisons.dropna(subset=["p_value"]).sort_values("p_value").iloc[0]
	example_stage = example["stage"]
	example_rows = summaries.loc[summaries["stage"] == example_stage].set_index(OUTCOME_COLUMN)
	correct_row = example_rows.loc["Correct"]
	incorrect_row = example_rows.loc["Incorrect"]
	direction = "lower" if example["rank_biserial_correct_minus_incorrect"] < 0 else "higher"
	significance = "statistically significant" if example["p_fdr_bh"] < alpha else "not statistically significant"
	lines.append(
		f"- **Numerical reading — {example_stage}:** Correct has median "
		f"{format_number(correct_row['median'])} (IQR {format_number(correct_row['q1'])}–{format_number(correct_row['q3'])}; "
		f"detected {int(correct_row['n_observed'])}/{int(correct_row['n_total'])}), while Incorrect has median "
		f"{format_number(incorrect_row['median'])} (IQR {format_number(incorrect_row['q1'])}–{format_number(incorrect_row['q3'])}; "
		f"detected {int(incorrect_row['n_observed'])}/{int(incorrect_row['n_total'])}). Correct detected layers tend to be "
		f"{direction} (rank-biserial = {format_number(example['rank_biserial_correct_minus_incorrect'], 2)}), but "
		f"$p_{{FDR-BH}}={format_number(example['p_fdr_bh'], 4)}$, so the contrast is {significance} at $\\alpha={alpha}$.")
	lines.append(
		f"> Among cases in which **{example_stage}** was detected, Correct and Incorrect queries showed "
		f"a descriptive tendency toward different layer locations (medians: "
		f"{format_number(correct_row['median'])} vs. {format_number(incorrect_row['median'])}; "
		f"rank-biserial = {format_number(example['rank_biserial_correct_minus_incorrect'], 2)}). "
		f"However, this contrast was {significance} after FDR correction across stages "
		f"($p_{{FDR-BH}}={format_number(example['p_fdr_bh'], 4)}$) and is therefore interpreted "
		f"{'as confirmatory evidence.' if example['p_fdr_bh'] < alpha else 'as descriptive rather than confirmatory evidence.'}"
	)
	lines.extend(["", "## Per-stage distribution summary", ""])

	for stage, stage_rows in summaries.groupby("stage", sort=False):
		lines.extend([f"### {stage}", ""])
		for _, row in stage_rows.iterrows():
			lines.append(
				f"- **{row[OUTCOME_COLUMN]}:** observed {int(row['n_observed'])}/{int(row['n_total'])} "
				f"({format_number(row['missing_percent'], 1)}% missing); median {format_number(row['median'])} "
				f"[Q1 {format_number(row['q1'])}, Q3 {format_number(row['q3'])}], IQR {format_number(row['iqr'])}; "
				f"whiskers {format_number(row['lower_whisker'])}–{format_number(row['upper_whisker'])}; "
				f"outliers {int(row['outlier_count'])}."
			)
		comparison = comparisons.loc[comparisons["stage"] == stage].iloc[0]
		if pd.notna(comparison.get("p_value")):
			significance = "significant" if comparison["p_fdr_bh"] < alpha else "not significant"
			direction = "higher" if comparison["median_difference_correct_minus_incorrect"] > 0 else "lower"
			lines.append(
				f"- **Correct vs Incorrect:** Correct median is {format_number(abs(comparison['median_difference_correct_minus_incorrect']))} layers {direction}; "
				f"$U={format_number(comparison['mann_whitney_u'], 1)}$, $p={format_number(comparison['p_value'], 4)}$, "
				f"$p_{{FDR-BH}}={format_number(comparison['p_fdr_bh'], 4)}$ ({significance} at $\\alpha={alpha}$), "
				f"rank-biserial = {format_number(comparison['rank_biserial_correct_minus_incorrect'], 2)}."
			)
		lines.append("")

	lines.extend(["## Median stage ordering", ""])
	for outcome, outcome_rows in summaries.groupby(OUTCOME_COLUMN, sort=False):
		ordered = outcome_rows.sort_values("median")
		ordering = " < ".join(
			f"{row['stage']} ({format_number(row['median'])})" for _, row in ordered.iterrows()
		)
		lines.append(f"- **{outcome}:** {ordering}")
	lines.append("")
	path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("csv_path", type=Path, help="Path to stages_boxplot.csv")
	parser.add_argument(
		"--output-dir", type=Path, default=None, help="Directory for reports (default: the CSV directory)"
	)
	parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold (default: 0.05)")
	args = parser.parse_args()

	if not 0 < args.alpha < 1:
		parser.error("--alpha must be between 0 and 1.")
	if not args.csv_path.is_file():
		parser.error(f"Input file does not exist: {args.csv_path}")

	data = pd.read_csv(args.csv_path)
	missing_columns = REQUIRED_COLUMNS - set(data.columns)
	if missing_columns:
		parser.error(f"Input is missing required columns: {', '.join(sorted(missing_columns))}")
	data["value"] = pd.to_numeric(data["value"], errors="coerce")

	summary_rows = []
	for (stage, outcome), values in data.groupby(["stage", OUTCOME_COLUMN], sort=False, dropna=False)["value"]:
		summary = {"stage": stage, OUTCOME_COLUMN: outcome}
		summary.update(boxplot_summary(values))
		summary_rows.append(summary)
	summaries = pd.DataFrame(summary_rows)
	comparisons = pd.DataFrame(
		[compare_outcomes(stage, group) for stage, group in data.groupby("stage", sort=False)]
	)
	comparisons["p_fdr_bh"] = benjamini_hochberg(comparisons.get("p_value", pd.Series(dtype=float)))

	output_dir = args.output_dir or args.csv_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)
	summary_path = output_dir / "stages_boxplot_statistics.csv"
	comparison_path = output_dir / "stages_boxplot_outcome_comparisons.csv"
	report_path = output_dir / "stages_boxplot_statistical_report.md"
	summaries.to_csv(summary_path, index=False, float_format="%.6f")
	comparisons.to_csv(comparison_path, index=False, float_format="%.6f")
	write_markdown_report(report_path, summaries, comparisons, args.alpha)

	print(f"Wrote distribution statistics: {summary_path}")
	print(f"Wrote outcome comparisons:    {comparison_path}")
	print(f"Wrote narrative report:       {report_path}")


if __name__ == "__main__":
	main()
