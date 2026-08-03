"""Create manifest-driven edit-statistics trajectory figures.

The output has one figure per metric.  Columns are models.  The top row shows
mean values by edited layer (with case-level bootstrap confidence intervals).
The bottom row shows the edit-request trajectory, smoothed by a configurable
rolling window.  All source paths, labels, and styles are recorded in a YAML
manifest so figures can be regenerated without clipboard or spreadsheet steps.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot edit-statistics trajectories from a YAML manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("edit_statistics_manifest.yaml"),
        help="YAML configuration file (default: edit_statistics_manifest.yaml beside this script).",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or "models" not in manifest or "metrics" not in manifest:
        raise ValueError("Manifest must define 'models' and 'metrics'.")
    return manifest


def load_records(manifest: Dict[str, Any]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model in manifest["models"]:
        model_label = model["label"]
        for method in model["methods"]:
            statistics_path = Path(method["path"]).resolve()
            if not statistics_path.is_file():
                raise FileNotFoundError(f"Missing edit-statistics file: {statistics_path}")
            with statistics_path.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError(f"Expected a list in {statistics_path}")
            for row in rows:
                record = dict(row)
                record["model_label"] = model_label
                record["method_label"] = method["label"]
                record["method_color"] = method.get("color")
                record["method_linestyle"] = method.get("linestyle", "-")
                record["method_linewidth"] = method.get("linewidth", 1.5)
                record["source_path"] = str(statistics_path)
                records.append(record)
    data = pd.DataFrame(records)
    required = {"model_label", "method_label", "edit_id", "edited_layer"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Statistics records are missing required fields: {sorted(missing)}")
    return data


def bootstrap_layer_intervals(values: np.ndarray, samples: int, seed: int) -> Tuple[float, float, float]:
    """Return mean and percentile-bootstrap CI for one method/layer group."""
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    boot_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(lower), float(upper)


def metric_subset(data: pd.DataFrame, metric: str, optimized_layers_only: bool) -> pd.DataFrame:
    """Return finite metric values, optionally only for target-optimization rows."""
    subset = data.dropna(subset=[metric]).copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    non_finite = ~np.isfinite(subset[metric])
    if non_finite.any():
        print(f"Excluding {non_finite.sum()} non-finite {metric} value(s).")
        subset = subset.loc[~non_finite]
    if optimized_layers_only:
        if "optimization_steps" not in subset.columns:
            raise ValueError(f"{metric} requires optimization_steps to select optimized layers.")
        subset = subset.loc[pd.to_numeric(subset["optimization_steps"], errors="coerce").fillna(0) > 0]
    return subset


def layer_summary(
    data: pd.DataFrame,
    metric: str,
    samples: int,
    seed: int,
    optimized_layers_only: bool,
) -> pd.DataFrame:
    rows = []
    subset = metric_subset(data, metric, optimized_layers_only)
    for group_index, ((model, method, layer), group) in enumerate(
        subset.groupby(["model_label", "method_label", "edited_layer"], sort=False)
    ):
        values = group[metric].to_numpy(dtype=float)
        mean, lower, upper = bootstrap_layer_intervals(values, samples, seed + group_index)
        first = group.iloc[0]
        rows.append({
            "model_label": model,
            "method_label": method,
            "edited_layer": layer,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "color": first["method_color"],
            "linestyle": first["method_linestyle"],
            "linewidth": first["method_linewidth"],
        })
    return pd.DataFrame(rows)


def edit_trajectory(
    data: pd.DataFrame,
    metric: str,
    rolling_window: int,
    aggregation: str,
    optimized_layers_only: bool,
) -> pd.DataFrame:
    """Aggregate layer values within edit requests, then apply a rolling mean."""
    subset = metric_subset(data, metric, optimized_layers_only)
    if aggregation == "final_optimized_layer":
        subset = (
            subset.sort_values(["model_label", "method_label", "edit_id", "edited_layer"])
            .groupby(["model_label", "method_label", "edit_id"], as_index=False, sort=False)
            .tail(1)
        )
        aggregation = "mean"
    if aggregation not in {"mean", "sum"}:
        raise ValueError(f"Unsupported trajectory aggregation '{aggregation}' for {metric}.")
    grouped = (
        subset.groupby(["model_label", "method_label", "edit_id"], as_index=False, sort=False)
        .agg(
            value=(metric, aggregation),
            color=("method_color", "first"),
            linestyle=("method_linestyle", "first"),
            linewidth=("method_linewidth", "first"),
        )
    )
    grouped = grouped.sort_values(["model_label", "method_label", "edit_id"])
    grouped["smoothed_value"] = (
        grouped.groupby(["model_label", "method_label"], sort=False)["value"]
        .transform(lambda values: values.rolling(rolling_window, center=True, min_periods=1).mean())
    )
    return grouped


def plot_metric(
    data: pd.DataFrame,
    manifest: Dict[str, Any],
    metric: Dict[str, str],
    output_directory: Path,
) -> None:
    metric_key = metric["key"]
    if metric_key not in data.columns:
        print(f"Skipping {metric_key}: not present in the configured statistics files.")
        return
    output = manifest.get("output", {})
    models = manifest["models"]
    model_labels = [model["label"] for model in models]
    samples = int(output.get("bootstrap_samples", 1000))
    seed = int(output.get("bootstrap_seed", 20260803))
    rolling_window = int(output.get("rolling_window", 25))
    yscale = metric.get("yscale", "linear")
    optimized_layers_only = bool(metric.get("optimized_layers_only", False))
    trajectory_aggregation = metric.get("trajectory", "mean")
    layer_data = layer_summary(data, metric_key, samples, seed, optimized_layers_only)
    trajectory_data = edit_trajectory(
        data, metric_key, rolling_window, trajectory_aggregation, optimized_layers_only
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=len(model_labels),
        figsize=(4.5 * len(model_labels), 6.6),
        squeeze=False,
        constrained_layout=True,
    )
    legend_handles: Dict[str, Line2D] = {}
    for column, model_label in enumerate(model_labels):
        top_axis, bottom_axis = axes[0, column], axes[1, column]
        top_axis.set_title(model_label)
        model_layers = layer_data[layer_data["model_label"] == model_label]
        model_trajectory = trajectory_data[trajectory_data["model_label"] == model_label]

        for method in models[column]["methods"]:
            label = method["label"]
            color = method.get("color")
            linestyle = method.get("linestyle", "-")
            linewidth = method.get("linewidth", 1.5)
            layer_rows = model_layers[model_layers["method_label"] == label].sort_values("edited_layer")
            if not layer_rows.empty:
                top_axis.plot(
                    layer_rows["edited_layer"], layer_rows["mean"], label=label,
                    color=color, linestyle=linestyle, linewidth=linewidth, marker="o", markersize=3,
                )
                top_axis.fill_between(
                    layer_rows["edited_layer"], layer_rows["lower"], layer_rows["upper"],
                    color=color, alpha=0.16,
                )
            trajectory_rows = model_trajectory[model_trajectory["method_label"] == label]
            if not trajectory_rows.empty:
                bottom_axis.plot(
                    trajectory_rows["edit_id"], trajectory_rows["smoothed_value"], label=label,
                    color=color, linestyle=linestyle, linewidth=linewidth,
                )
            legend_handles.setdefault(
                label,
                Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=label),
            )

        top_axis.set_xlabel("Edited layer")
        top_axis.set_xticks(sorted(model_layers["edited_layer"].unique()))
        bottom_axis.set_xlabel("Edit request ID")
        for axis in (top_axis, bottom_axis):
            axis.set_yscale(yscale)
            axis.grid(axis="y", alpha=0.3)
            axis.margins(x=0.02)
        if column == 0:
            scale_suffix = " (log scale)" if yscale == "log" else ""
            top_axis.set_ylabel(f"{metric['label']}{scale_suffix}\nmean ± 95% bootstrap CI")
            aggregation_label = {
                "sum": "sum across layers",
                "final_optimized_layer": "final optimized layer",
                "mean": "mean across layers",
            }[trajectory_aggregation]
            bottom_axis.set_ylabel(
                f"{metric['label']}{scale_suffix}\n{rolling_window}-edit rolling mean ({aggregation_label})"
            )

    figure.legend(
        handles=list(legend_handles.values()),
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    for file_format in output.get("formats", ["pdf"]):
        destination = output_directory / f"{metric['filename']}.{file_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if file_format.lower() not in {"pdf", "svg"}:
            save_kwargs["dpi"] = output.get("dpi", 300)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(destination, **save_kwargs)
        print(f"Saved {destination}")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    data = load_records(manifest)
    output_directory = Path(manifest.get("output", {}).get("directory", "figures")).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    for metric in manifest["metrics"]:
        plot_metric(data, manifest, metric, output_directory)


if __name__ == "__main__":
    main()
