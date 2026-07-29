"""Generate figure-independent statistics for entity-description heatmaps.

Input CSVs are written by ``analysis.analyze_results.plot_layer_matrix`` and
contain a percentage for every (source_layer, target_layer) cell. Each cell is
the percentage of examples, for that source/target layer intervention, whose
generated entity description is classified as the heatmap target. Therefore,
rows and columns are separate experimental conditions, not a joint probability
distribution; their totals should not be interpreted as percentages of one
common population.

Example
-------
python entity_description_to_statistics.py entity_description_e1_e2.csv \
    entity_description_last_e2.csv entity_description_last_e3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"source_layer", "target_layer", "percent"}


def label_from_path(path: Path) -> str:
	"""Use a concise heatmap name in reports."""
	return path.stem.removeprefix("entity_description_")


def entropy_concentration(values: np.ndarray) -> float:
	"""Return normalized entropy-based concentration from 0 (diffuse) to 1 (focused)."""
	positive = values[values > 0]
	if len(positive) <= 1:
		return 1.0 if len(positive) else np.nan
	probabilities = positive / positive.sum()
	return 1 - (-np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities)))


def weighted_layer_mean(totals: pd.Series) -> float:
	"""Return an average layer weighted by response mass, or NA if no response exists."""
	return np.average(totals.index.to_numpy(dtype=float), weights=totals.to_numpy()) if totals.sum() else np.nan


def top_items(totals: pd.Series, count: int = 3) -> str:
	"""Format the highest-mass layers as layer (percentage) text."""
	return "; ".join(f"{int(layer)} ({value:.2f}%)" for layer, value in totals.sort_values(ascending=False).head(count).items())


def summarize_heatmap(label: str, data: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
	"""Calculate heatmap-wide, row, and column statistics."""
	matrix = data.pivot(index="source_layer", columns="target_layer", values="percent").fillna(0).sort_index().sort_index(axis=1)
	values = matrix.to_numpy(dtype=float)
	source, target = np.meshgrid(matrix.index.to_numpy(), matrix.columns.to_numpy(), indexing="ij")
	max_value = values.max()
	max_cells = np.argwhere(np.isclose(values, max_value))
	all_max_locations = [f"({int(matrix.index[row])}, {int(matrix.columns[column])})" for row, column in max_cells]
	max_locations = "; ".join(all_max_locations[:5])
	if len(all_max_locations) > 5:
		max_locations += "; ..."
	diagonal = np.diag(matrix.reindex(index=matrix.index, columns=matrix.index, fill_value=0).to_numpy())
	diagonal_mass = diagonal.sum()
	total_mass = values.sum()
	below_mass = values[target < source].sum()
	above_mass = values[target > source].sum()
	nonzero = values[values > 0]
	row_totals, column_totals = matrix.sum(axis=1), matrix.sum(axis=0)

	summary = {
		"heatmap": label,
		"source_layer_count": len(matrix.index),
		"target_layer_count": len(matrix.columns),
		"cell_count": values.size,
		"nonzero_cell_count": int((values > 0).sum()),
		"nonzero_cell_percent": 100 * (values > 0).mean(),
		"mean_cell_percent": values.mean(),
		"median_cell_percent": np.median(values),
		"max_cell_percent": max_value,
		"max_cell_count": len(max_cells),
		"max_cell_locations_source_target": max_locations,
		"total_response_mass": total_mass,
		"diagonal_mass_percent_of_total": 100 * diagonal_mass / total_mass if total_mass else np.nan,
		"below_diagonal_mass_percent_of_total": 100 * below_mass / total_mass if total_mass else np.nan,
		"above_diagonal_mass_percent_of_total": 100 * above_mass / total_mass if total_mass else np.nan,
		"response_weighted_source_layer": np.average(source, weights=values) if total_mass else np.nan,
		"response_weighted_target_layer": np.average(target, weights=values) if total_mass else np.nan,
		"target_minus_source_weighted_layers": np.average(target - source, weights=values) if total_mass else np.nan,
		"cell_mass_concentration": entropy_concentration(nonzero),
		"top_source_layers_by_row_mass": top_items(row_totals),
		"top_target_layers_by_column_mass": top_items(column_totals),
	}

	row_stats = pd.DataFrame({
		"heatmap": label,
		"source_layer": row_totals.index,
		"row_response_mass": row_totals.values,
		"row_peak_percent": matrix.max(axis=1).values,
		"row_peak_target_layers": [
			", ".join(str(int(layer)) for layer in matrix.columns[np.isclose(matrix.loc[index], matrix.loc[index].max())])
			for index in matrix.index
		],
		"target_layer_weighted_by_row_response": [weighted_layer_mean(matrix.loc[index]) for index in matrix.index],
	})
	column_stats = pd.DataFrame({
		"heatmap": label,
		"target_layer": column_totals.index,
		"column_response_mass": column_totals.values,
		"column_peak_percent": matrix.max(axis=0).values,
		"column_peak_source_layers": [
			", ".join(str(int(layer)) for layer in matrix.index[np.isclose(matrix[column], matrix[column].max())])
			for column in matrix.columns
		],
		"source_layer_weighted_by_column_response": [weighted_layer_mean(matrix[column]) for column in matrix.columns],
	})
	return summary, row_stats, column_stats


def number(value: object, digits: int = 2) -> str:
	return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def write_report(path: Path, summaries: pd.DataFrame) -> None:
	"""Write an interpretable narrative for all supplied heatmaps."""
	lines = [
		"# Entity-description heatmap statistical report",
		"",
		"Each cell is the percentage of examples classified as the target entity under one source-layer / target-layer condition. Heatmap cells are condition-specific rates: summing cells, rows, or columns does **not** produce an overall success percentage.",
		"",
		"## How to read the statistics",
		"",
		"- **Axis convention:** The CSV defines `(source_layer, target_layer)` as `(row, column)`. Direction is always computed numerically as `target − source`, irrespective of the heatmap renderer's visual origin: `target < source` is below diagonal, and `target > source` is above diagonal.",
		"- **Cell percentage / maximum cell:** Compute this to identify the strongest individual source–target condition. For 12.50% at (22, 1), patching source layer 22 and evaluating target layer 1 yields target-entity classification in 12.50% of examples under that condition.",
		"- **Nonzero-cell coverage:** Compute this to distinguish focal from widespread effects. For 137/1024 nonzero cells (13.38%), only a small subset of tested layer-pair conditions produces any target-entity classification.",
		"- **Row / column response mass:** Compute these to rank source and target layers by their aggregate response across the tested conditions. A row mass of 66.70% for source layer 24 is the sum of its 32 condition-specific cell rates, so it is a relative importance score—not a 66.70% success probability.",
		"- **Diagonal / direction mass:** Compute these shares to describe the heatmap geometry: diagonal cells have equal source and target layers; cells with target < source are below diagonal; cells with target > source are above diagonal. For below/diagonal/above shares of 87.34%/3.80%/8.86%, most response intensity lies in conditions with earlier target layers; this is descriptive, not causal or a significance test.",
		"- **Response-weighted layer gap:** Compute the intensity-weighted mean of target − source to quantify the magnitude of the directional pattern. A gap of −10.74 means high-response cells have target layers about 10.74 layers earlier than source layers on average; positive values instead indicate later targets.",
		"- **Cell-mass concentration:** Compute this 0–1 entropy-based index to distinguish a few dominant hotspots (near 1) from similarly sized positive cells spread across a region (near 0). A value of 0.01 means the positive cell intensities are comparatively even; it does not mean that the positive cells are geometrically dispersed or that many cells are nonzero.",
		"- **Why aggregates can differ:** Top row/column masses identify which source/target layers participate most often, whereas the weighted gap uses every cell's signed target − source offset and concentration uses the relative balance among positive cells; neither statistic can be inferred from the top three rows or columns alone.",
		"",
		"## Heatmap summaries",
		"",
	]
	for _, row in summaries.iterrows():
		direction = "later target layers" if row["target_minus_source_weighted_layers"] > 0 else "earlier target layers"
		lines.extend([
			f"### {row['heatmap']}",
			"",
			f"- **Coverage and peak:** {int(row['nonzero_cell_count'])}/{int(row['cell_count'])} cells are nonzero ({number(row['nonzero_cell_percent'])}%). The maximum cell is {number(row['max_cell_percent'])}% and occurs in {int(row['max_cell_count'])} cell(s): {row['max_cell_locations_source_target']}.",
			f"- **Structure:** Of the total response mass, {number(row['diagonal_mass_percent_of_total'])}% is diagonal, {number(row['above_diagonal_mass_percent_of_total'])}% is above diagonal, and {number(row['below_diagonal_mass_percent_of_total'])}% is below diagonal. The weighted layer gap is {number(row['target_minus_source_weighted_layers'])}, indicating a tendency toward {direction}.",
			f"- **Concentration:** cell-mass concentration = {number(row['cell_mass_concentration'])}; top source layers by row-response mass: {row['top_source_layers_by_row_mass']}; top target layers by column-response mass: {row['top_target_layers_by_column_mass']}.",
			"",
		])

	if not summaries.empty:
		example = summaries.loc[summaries["max_cell_percent"].idxmax()]
		lines.extend([
			"## Worked example and paper-ready interpretation",
			"",
			f"- **Numerical reading — {example['heatmap']}:** Its strongest tested condition is {number(example['max_cell_percent'])}% in {int(example['max_cell_count'])} cell(s), including {example['max_cell_locations_source_target']}. Positive responses occur in {int(example['nonzero_cell_count'])}/{int(example['cell_count'])} cells ({number(example['nonzero_cell_percent'])}%), so the response is {'focal' if example['nonzero_cell_percent'] < 25 else 'distributed'} across tested layer pairs.",
			f"> In the **{example['heatmap']}** heatmap, target-entity classification reached a maximum of {number(example['max_cell_percent'])}% in {int(example['max_cell_count'])} source–target layer condition(s), including {example['max_cell_locations_source_target']}. The response occupied {number(example['nonzero_cell_percent'])}% of tested layer pairs and showed a response-weighted target-minus-source gap of {number(example['target_minus_source_weighted_layers'])} layers. This is a descriptive localization pattern across intervention conditions and does not by itself establish a causal information-flow mechanism.",
			"",
		])
	path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("csv_paths", nargs="+", type=Path, help="One or more entity_description_*.csv heatmap files")
	parser.add_argument("--output-dir", type=Path, default=None, help="Directory for reports (default: first CSV directory)")
	args = parser.parse_args()
	for csv_path in args.csv_paths:
		if not csv_path.is_file():
			parser.error(f"Input file does not exist: {csv_path}")

	summaries, all_rows, all_columns = [], [], []
	for csv_path in args.csv_paths:
		data = pd.read_csv(csv_path)
		missing_columns = REQUIRED_COLUMNS - set(data.columns)
		if missing_columns:
			parser.error(f"{csv_path.name} is missing required columns: {', '.join(sorted(missing_columns))}")
		for column in REQUIRED_COLUMNS:
			data[column] = pd.to_numeric(data[column], errors="coerce") if column != "percent" or data[column].dtype != object else pd.to_numeric(data[column], errors="coerce")
		if data[list(REQUIRED_COLUMNS)].isna().any().any() or (data["percent"] < 0).any():
			parser.error(f"{csv_path.name} requires non-missing numeric layers/percentages and non-negative percentages.")
		if data.duplicated(["source_layer", "target_layer"]).any():
			parser.error(f"{csv_path.name} contains duplicate source/target layer cells.")
		summary, row_stats, column_stats = summarize_heatmap(label_from_path(csv_path), data)
		summaries.append(summary)
		all_rows.append(row_stats)
		all_columns.append(column_stats)

	summary_frame = pd.DataFrame(summaries)
	output_dir = args.output_dir or args.csv_paths[0].parent
	output_dir.mkdir(parents=True, exist_ok=True)
	summary_path = output_dir / "entity_description_heatmap_statistics.csv"
	row_path = output_dir / "entity_description_heatmap_row_statistics.csv"
	column_path = output_dir / "entity_description_heatmap_column_statistics.csv"
	report_path = output_dir / "entity_description_heatmap_statistical_report.md"
	summary_frame.to_csv(summary_path, index=False, float_format="%.6f")
	pd.concat(all_rows, ignore_index=True).to_csv(row_path, index=False, float_format="%.6f")
	pd.concat(all_columns, ignore_index=True).to_csv(column_path, index=False, float_format="%.6f")
	write_report(report_path, summary_frame)
	print(f"Wrote heatmap statistics: {summary_path}")
	print(f"Wrote row statistics:     {row_path}")
	print(f"Wrote column statistics:  {column_path}")
	print(f"Wrote narrative report:   {report_path}")


if __name__ == "__main__":
	main()
