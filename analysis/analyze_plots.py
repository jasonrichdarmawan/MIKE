# %%
"""
Generate mechanism plots from layers.csv and entity description results.

Works for both pre-edit and post-edit subsets — pass any number of named
directory paths via --subsets.  Each directory must contain layers.csv
(or source files from which it can be built) and entity_description/.

Produces:
  stages_boxplot.pdf              -- stage layer distribution per subset
  min_entity_description_layers.pdf -- % cases where e2/e3 decoded per layer
  entity_description_e1_e2.pdf   -- PatchScopes heatmap: e1 source, e2 target
  entity_description_last_e2.pdf -- PatchScopes heatmap: last source, e2 target
  entity_description_last_e3.pdf -- PatchScopes heatmap: last source, e3 target

Usage:
    # Pre-edit
    python analysis/analyze_plots.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --output-dir plots/meta-llama/Meta-Llama-3-8B \\
        --subsets \\
            correct:datasets/meta-llama/Meta-Llama-3-8B/composition_correct \\
            incorrect:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect

    # Pre- and post-edit comparison
    python analysis/analyze_plots.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --output-dir plots/meta-llama/Meta-Llama-3-8B/post_alphaedit \\
        --subsets \\
            pre-edit:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect \\
            now_correct:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/now_correct \\
            still_incorrect:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/still_incorrect
"""

# %%

import ast
import sys
sys.path.insert(0, "analysis")

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from classify_results import load_entity_description_generations, classify_generations, classify_entity

tqdm.pandas()
sns.set_theme()
sns.set_style("whitegrid")


# %% Loaders

def load_layers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    for col in df.columns:
        if col not in ["id", "composition_correct"]:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) is not float else x)
    return df


# %% Helpers

def save_plot(path: str, title=None, xlabel: str = None, ylabel: str = None, legend=True):
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()
    if title is not None:
        plt.title(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_matrix(matrix: np.ndarray, labels: pd.Index, x_label: str, y_label: str, vmax=None):
    with plt.rc_context(rc={"font.size": 60}):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        im = ax.imshow(matrix, origin="lower", vmin=0, vmax=vmax) if vmax else ax.imshow(matrix, origin="lower")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ticks = np.arange(0, len(labels), 8)
        ax.set_xticks(ticks, labels=[labels[i] for i in ticks])
        ax.set_yticks(ticks, labels=[labels[i] for i in ticks])


# %% Plot: stage boxplots

def plot_stage_boxplots(subsets: dict, path: str):
    """
    Boxplot of stage layer distributions across subsets.

    Stages:
      e1_e2_entity_layer          — 1st Hop Resolved
      info_prop                   — Info Propagation (mean of knockout + attn projection + patchscopes)
      last_e3_entity_layer        — 2nd Hop Resolved
      mlp_prediction_projection_layers — Prediction Extracted
    """
    stage_cols = [
        "e1_e2_entity_layer", "last_e1_attention_knockout_layers",
        "attention_e2_projection_layers", "last_e2_entity_layer",
        "last_e3_entity_layer", "mlp_prediction_projection_layers",
    ]
    labels = {
        "e1_e2_entity_layer":                  "1st Hop Resolved",
        "info_prop":                            "Info Propagation",
        "last_e3_entity_layer":                 "2nd Hop Resolved",
        "mlp_prediction_projection_layers":     "Prediction Extracted",
    }

    records = []
    for name, df in subsets.items():
        df = df.copy()
        for col in stage_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: min(x) if isinstance(x, list) and x else np.nan)
        df["info_prop"] = df.apply(lambda row: np.nanmean([
            row.get("last_e1_attention_knockout_layers", np.nan),
            row.get("attention_e2_projection_layers", np.nan),
            row.get("last_e2_entity_layer", np.nan),
        ]), axis=1)
        melted = pd.melt(df, value_vars=list(labels.keys()), value_name="value", var_name="stage")
        melted["subset"] = name
        records.append(melted)

    combined = pd.concat(records).reset_index(drop=True)
    combined["stage"] = combined["stage"].map(labels)

    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        sns.boxplot(x="stage", y="value", data=combined, hue="subset", gap=0.1, ax=ax)
        ax.set_xlabel(None)
        ax.set_xticklabels(labels=list(labels.values()), rotation=45, ha="right", rotation_mode="anchor")
        sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.2),
                        ncol=min(len(subsets), 3), title="Subset")
        save_plot(path, ylabel="Layer", legend=False)


# %% Plot: min entity description layers

def plot_min_entity_description_layers(subsets: dict, path: str, n_layers: int):
    """
    For each subset: % of cases where e2 (from t1) and e3 (from t2) are decoded
    at each layer (minimum layer across patchscopes source layers).
    """
    def _pct_at_layer(series, n_layers):
        mins = series.apply(lambda x: min(x) if isinstance(x, list) and x else np.nan)
        return mins.value_counts(normalize=True, dropna=True).sort_index().reindex(
            range(n_layers), fill_value=0) * 100

    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        linestyles = ["-", "--", ":", "-."]
        for i, (name, df) in enumerate(subsets.items()):
            ls = linestyles[i % len(linestyles)]
            if "e1_e2_entity_layer" in df.columns:
                data = _pct_at_layer(df["e1_e2_entity_layer"], n_layers)
                sns.lineplot(data=data, ax=ax, label=f"{name} / e2 from t1", linestyle=ls)
            if "last_e3_entity_layer" in df.columns:
                data = _pct_at_layer(df["last_e3_entity_layer"], n_layers)
                sns.lineplot(data=data, ax=ax, label=f"{name} / e3 from t2", linestyle="--")
        sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                        ncol=2, fontsize=8)
        save_plot(path, xlabel="Layer", ylabel="Percentage of Cases", legend=False)


# %% Plot: entity description heatmaps

def plot_layer_matrix(generations: pd.DataFrame, target_classification: str, path: str):
    success_matrix = (
        generations.groupby(["source_layer", "target_layer"])["classification"]
        .value_counts(normalize=True)
        .unstack("classification")
        .fillna(0) * 100
    )
    if target_classification not in success_matrix.columns:
        print(f"  [SKIP] {path}: classification '{target_classification}' not found")
        return
    matrix = success_matrix[target_classification].unstack("target_layer").fillna(0)
    plot_matrix(matrix.to_numpy(), matrix.columns, "Target Layer", "Source Layer")
    save_plot(path, legend=True)


# %% Main

def main(args):
    layer_count = {
        "gpt2": 12,
        "meta-llama/Llama-2-7b-hf": 32,
        "meta-llama/Llama-2-13b-hf": 40,
        "meta-llama/Meta-Llama-3-8B": 32,
        "meta-llama/Meta-Llama-3-70B": 80,
    }
    n_layers   = layer_count[args.model]
    plots_path = args.output_dir

    # ── Load subsets ──────────────────────────────────────────────────────────
    subsets = {}
    for entry in args.subsets:
        name, path = entry.split(":", 1)
        layers_csv = Path(path) / "layers.csv"
        if not layers_csv.exists():
            print(f"[SKIP] {name}: {layers_csv} not found — run analyze_tables.py first")
            continue
        subsets[name] = load_layers(str(layers_csv))

    if not subsets:
        print("No valid subsets loaded.")
        return

    # ── Stage boxplots ────────────────────────────────────────────────────────
    print("Plotting stage boxplots ...")
    plot_stage_boxplots(subsets, f"{plots_path}/stages_boxplot.pdf")
    print(f"Saved -> {plots_path}/stages_boxplot.pdf")

    # ── Min entity description layers ─────────────────────────────────────────
    print("Plotting min entity description layers ...")
    plot_min_entity_description_layers(subsets, f"{plots_path}/min_entity_description_layers.pdf", n_layers)
    print(f"Saved -> {plots_path}/min_entity_description_layers.pdf")

    # ── Entity description heatmaps (first subset that has data) ─────────────
    print("Plotting entity description heatmaps ...")
    for name, path in {e.split(":", 1)[0]: e.split(":", 1)[1] for e in args.subsets}.items():
        for source, target, fname in [
            ("e1",   "e2", "entity_description_e1_e2.pdf"),
            ("last", "e2", "entity_description_last_e2.pdf"),
            ("last", "e3", "entity_description_last_e3.pdf"),
        ]:
            csv_path = f"{path}/entity_description/{source}_entity_description_s0.csv"
            try:
                gens = classify_generations(csv_path, classify_entity)
            except FileNotFoundError:
                continue
            out = f"{plots_path}/{name}_{fname}"
            plot_layer_matrix(gens, target, out)
            print(f"Saved -> {out}")


# %%

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", required=True,
                        metavar="NAME:PATH",
                        help="One or more name:path pairs where path is the subset directory "
                             "containing layers.csv and entity_description/")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write plot PDFs into")
    parser.add_argument("-m", "--model", required=True,
                        choices=["gpt2",
                                 "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
                                 "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"],
                        help="Model name (used for layer count metadata only)")
    main(parser.parse_args())
