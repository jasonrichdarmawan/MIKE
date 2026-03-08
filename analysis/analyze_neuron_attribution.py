"""
Analyze neuron-level attribution results (LPI and residual stream decomposition) across subsets.

Two types of scores are used throughout:

  LPI (Log Probability Increase):
    How much a sublayer/head/neuron increases log P(predicted token) at the last position.
    Unit: nats (direct contribution to final prediction).
    Source: neuron_lpi.py -> sublayer_lpi.csv, head_lpi.csv, ffn_neurons.csv, attn_neurons.csv.

  LPI-weighted residual contribution (query score):
    For a prior FFN neuron activating a top-K value neuron:
        score = (inner_product(neuron_output, V_key) / total_residual_alignment) * value_neuron_LPI
    Unit: fraction of residual stream alignment with the V-key vector, weighted by the
    downstream value neuron's LPI. Captures indirect upstream influence, not direct LPI.
    Source: neuron_query.py -> ffn_query_layers.csv, attn_query_layers.csv, attn_query_neurons.csv.

Produces:
  sublayer_lpi_profile.pdf    -- mean LPI per layer as lines (correct/incorrect/inconsistent)
  sublayer_lpi_bars.pdf       -- interleaved attention/FFN bars per layer, one panel per subset
                                 (mirrors GPT2_view_knowledge.ipynb cell 5)
  head_lpi_heatmap.pdf        -- mean head LPI heatmap (layer x head) per subset
  neuron_layer_dist.pdf       -- top-K FFN and attn neuron counts per layer, one line per subset
                                 (mirrors GPT2_view_knowledge.ipynb cells 10 and 12)
  query_layer_bars.pdf        -- residual stream decomposition: LPI-weighted contribution of
                                 prior layers to top-K value neurons (requires neuron_query.py)
                                 (mirrors GPT2_view_knowledge.ipynb cells 13 and 14)
  query_value_neurons.pdf     -- queryonly vs queryvalue: LPI-weighted upstream influence
                                 of prior FFN neurons, with and without overlap filter
                                 (mirrors GPT2_view_knowledge.ipynb cell 17)

Cross-method comparisons (LPI vs projection, knockout, patching) are in compare_methods.py.

Usage:
    python analyze_neuron_attribution.py <datasets_dir> <output_path> -m gpt2
"""

from argparse import ArgumentParser
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme()
sns.set_style("whitegrid")

SUBSETS = ["composition_correct", "composition_incorrect", "composition_inconsistent"]
SUBSET_LABELS = ["correct", "incorrect", "inconsistent"]

PROMPT_SUBDIR = {
    "composition": "neuron_attribution",
    "first-hop":   "neuron_attribution_first_hop",
    "second-hop":  "neuron_attribution_second_hop",
}


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_neuron_lpi(dataset_dir: str, attr_subdir: str = "neuron_attribution"):
    path = Path(dataset_dir) / attr_subdir
    if not path.exists():
        raise FileNotFoundError(f"No neuron attribution directory found: {path}")

    def _optional(fname):
        p = path / fname
        return pd.read_csv(p) if p.exists() else None

    return {
        "sublayer":           pd.read_csv(path / "sublayer_lpi.csv"),
        "head":               pd.read_csv(path / "head_lpi.csv"),
        "ffn_neurons":        pd.read_csv(path / "ffn_neurons.csv"),
        "attn_neurons":       pd.read_csv(path / "attn_neurons.csv"),
        "ffn_query_layers":   _optional("ffn_query_layers.csv"),
        "attn_query_layers":  _optional("attn_query_layers.csv"),
        "attn_query_neurons": _optional("attn_query_neurons.csv"),
    }


# ── Plot 1: sublayer LPI line profile (subset comparison) ────────────────────

def plot_sublayer_lpi_profile(lpi_data, n_layers: int, path: str):
    """
    Two-panel line plot (attention | FFN), mean LPI per layer with 95% CI,
    one line per subset. Best for comparing correct vs incorrect vs inconsistent.
    """
    records = []
    for label, df in lpi_data.items():
        d = df[["id", "layer", "sublayer", "lpi"]].copy()
        d["subset"] = label
        records.append(d)
    combined = pd.concat(records)

    with sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, sublayer in zip(axes, ["attention", "ffn"]):
            data = combined[combined["sublayer"] == sublayer]
            sns.lineplot(data=data, x="layer", y="lpi", hue="subset", ax=ax)
            ax.set_title(sublayer.capitalize())
            ax.set_xlabel("Layer")
            ax.set_xticks(range(0, n_layers, 2))
            if sublayer == "attention":
                ax.set_ylabel("Mean LPI")
                ax.get_legend().remove()
            else:
                ax.set_ylabel("")
        sns.move_legend(axes[1], loc="upper left")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 2: interleaved attention/FFN bars (mirrors notebook plt_bar) ─────────

def plot_sublayer_lpi_bars(lpi_data, n_layers: int, path: str):
    """
    One panel per subset. At each layer, two adjacent bars:
      - darksalmon  = attention mean LPI
      - lightseagreen = FFN mean LPI
    Mirrors GPT2_view_knowledge.ipynb cell 5 (plt_bar), extended to population means.

    Caveat:
    - Error bars show +/- 1 standard error across examples.
    - Negative LPI (suppressive layers) is shown naturally below 0.
    """
    n = len(lpi_data)
    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 3.5), sharey=True)
        if n == 1:
            axes = [axes]

        bar_w = 0.4
        x = np.arange(n_layers)

        for ax, (label, df) in zip(axes, lpi_data.items()):
            stats = df.groupby(["layer", "sublayer"])["lpi"].agg(["mean", "sem"]).reset_index()
            attn = stats[stats["sublayer"] == "attention"].set_index("layer").reindex(range(n_layers))
            ffn  = stats[stats["sublayer"] == "ffn"].set_index("layer").reindex(range(n_layers))

            ax.bar(x - bar_w / 2, attn["mean"], bar_w,
                   yerr=attn["sem"], color="darksalmon", label="attention", capsize=2)
            ax.bar(x + bar_w / 2, ffn["mean"],  bar_w,
                   yerr=ffn["sem"],  color="lightseagreen", label="FFN", capsize=2)

            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(label.capitalize())
            ax.set_xlabel("Layer")
            ax.set_xticks(range(0, n_layers, 2))
            if ax == axes[0]:
                ax.set_ylabel("Mean LPI")

        axes[-1].legend(loc="upper left")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 3: head LPI heatmap ─────────────────────────────────────────────────

def plot_head_lpi_heatmap(lpi_data, n_layers: int, n_heads: int, path: str):
    """
    Mean head LPI heatmap (layer x head) per subset, shared colour scale.
    Mirrors GPT2_view_knowledge.ipynb cell 6 (plt_heatmap), extended to population means.
    """
    vmax = max(
        df.groupby(["layer", "head"])["lpi"].mean().max()
        for df in lpi_data.values()
    )
    n = len(lpi_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, lpi_data.items()):
        matrix = (
            df.groupby(["layer", "head"])["lpi"].mean()
            .unstack("head")
            .reindex(index=range(n_layers), columns=range(n_heads), fill_value=0)
        )
        im = ax.imshow(matrix.values, origin="lower", vmin=0, vmax=vmax,
                       aspect="auto", cmap="hot_r")
        ax.set_title(label.capitalize())
        ax.set_xlabel("Head")
        ax.set_xticks(range(0, n_heads, max(1, n_heads // 6)))
        ax.set_yticks(range(n_layers))

    axes[0].set_ylabel("Layer")
    fig.colorbar(im, ax=axes[-1], orientation="vertical", label="Mean LPI")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ── Plot 4: neuron layer distribution (mirrors notebook cells 10 and 12) ─────

def plot_neuron_layer_distribution(ffn_data, attn_data, n_layers: int, path: str):
    """
    For each subset, count the average number of top-K neurons per layer per example.
    Two panels: FFN neurons (left) | Attn neurons (right).
    Mirrors GPT2_view_knowledge.ipynb cells 10 and 12.

    Caveat:
    - 'top-K' here means all neurons stored in the CSV (already filtered to top-K per
      example by neuron_lpi.py). The y-axis is avg neuron count per layer per example,
      showing which layers dominate the top-K ranking across the dataset.
    """
    with sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, (data_dict, title, color_key) in zip(
            axes,
            [(ffn_data,  "FFN value neurons",  "bo-"),
             (attn_data, "Attn value neurons", "ro-")],
        ):
            for label, df in data_dict.items():
                n_examples = df["id"].nunique()
                counts = (
                    df.groupby("layer")["id"].count() / n_examples
                ).reindex(range(n_layers), fill_value=0)
                ax.plot(range(n_layers), counts.values, label=label)

            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Avg neuron count per example")
            ax.set_xticks(range(0, n_layers, 2))
            ax.legend(loc="upper left")

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 5: query layer bars (mirrors notebook cells 13-14) ──────────────────

def plot_query_layer_bars(ffn_query_data, attn_query_data, n_layers, path):
    """
    Two-panel line plot: FFN query (left) | attn query (right).
    Each panel shows mean LPI-weighted residual contribution per layer, with
    separate lines for attention vs FFN sublayer contributions, one line style
    per subset.

    Score = (inner_product_fraction of residual with V/FFN key) * value_neuron_LPI.
    Positive = excitatory contribution; negative = suppressive.

    Mirrors GPT2_view_knowledge.ipynb cells 13 (ffn_query_layers) and 14
    (attn_query_layers).

    Caveat:
    - Only subsets with query data (non-None) are plotted.
    - ffn_query_data / attn_query_data: dict label -> DataFrame or None.
    """
    available_ffn  = {k: v for k, v in ffn_query_data.items()  if v is not None}
    available_attn = {k: v for k, v in attn_query_data.items() if v is not None}
    if not available_ffn and not available_attn:
        print("  [SKIP] plot_query_layer_bars: no query layer data available")
        return

    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
        bar_w = 0.35
        x = np.arange(n_layers)

        for ax, data_dict, title in [
            (axes[0], available_ffn,  "FFN value neurons - query layers"),
            (axes[1], available_attn, "Attn value neurons - query layers"),
        ]:
            for label, df in data_dict.items():
                stats = df.groupby(["layer", "sublayer"])["score"].mean().unstack("sublayer")
                attn_s = stats.get("attention", pd.Series(dtype=float)).reindex(range(n_layers), fill_value=0)
                ffn_s  = stats.get("ffn",       pd.Series(dtype=float)).reindex(range(n_layers), fill_value=0)
                ax.plot(x - bar_w / 2, attn_s.values, "o-",  label=f"{label}/attn")
                ax.plot(x + bar_w / 2, ffn_s.values,  "s--", label=f"{label}/ffn")

            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_ylabel("LPI-weighted residual contribution")
            ax.set_xticks(range(0, n_layers, 2))
            ax.legend(loc="upper left", fontsize=7)

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 6: query value neurons (mirrors notebook cell 17) ────────────────────

def plot_query_value_neurons(attn_query_neuron_data, ffn_neuron_data, n_layers, path):
    """
    Two-panel plot: upstream influence of prior FFN neurons on top-K attn value neurons.

    Score = (inner_product(coeff * down_proj_col, V_key) / total_residual) * attn_neuron_LPI.
    This is an LPI-weighted residual contribution, NOT a direct LPI measurement.

    Left panel  (queryonly):  mean score by layer for all neurons in attn_query_neurons.csv
                              — which prior layers most activate the high-LPI attn neurons.
    Right panel (queryvalue): same but restricted to neurons that also appear in
                              ffn_neurons.csv — neurons that are BOTH high-LPI direct
                              contributors AND strong upstream activators.

    Mirrors GPT2_view_knowledge.ipynb cell 17: queryonly vs queryvalue.

    Caveat:
    - Only subsets with attn_query_neurons data (non-None) are plotted.
    - Overlap is computed per-example: a neuron is a value neuron if it appears
      in ffn_neurons.csv for the same example id.
    - attn_query_neuron_data / ffn_neuron_data: dict label -> DataFrame or None.
    """
    available = {k: v for k, v in attn_query_neuron_data.items() if v is not None}
    if not available:
        print("  [SKIP] plot_query_value_neurons: no attn_query_neurons data available")
        return

    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

        for label, qn_df in available.items():
            # queryonly: mean score per layer across all query neurons
            queryonly = (
                qn_df.groupby("layer")["score"].mean()
                .reindex(range(n_layers), fill_value=0)
            )
            axes[0].plot(range(n_layers), queryonly.values, "o-", label=label)

            # queryvalue: restrict to neurons that are also high-LPI FFN value neurons
            ffn_df = ffn_neuron_data.get(label)
            if ffn_df is not None:
                value_set = set(zip(ffn_df["id"], ffn_df["layer"], ffn_df["neuron_id"]))
                mask = qn_df.apply(
                    lambda r: (r["id"], r["layer"], r["neuron_id"]) in value_set, axis=1
                )
                overlap_df = qn_df[mask]
                if len(overlap_df) > 0:
                    queryvalue = (
                        overlap_df.groupby("layer")["score"].mean()
                        .reindex(range(n_layers), fill_value=0)
                    )
                    axes[1].plot(range(n_layers), queryvalue.values, "s--", label=label)

        for ax, title in [
            (axes[0], "Query neurons by layer (all)"),
            (axes[1], "Query-value overlap by layer"),
        ]:
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_ylabel("LPI-weighted residual contribution")
            ax.set_xticks(range(0, n_layers, 2))
            ax.legend(loc="upper left")

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 7: neuron overlap across prompt types ────────────────────────────────

def _jaccard_per_layer(df_a, df_b, id_cols, n_layers):
    """
    Mean Jaccard similarity per layer between two neuron DataFrames.

    For each (example, layer), computes |intersection| / |union| of neuron sets
    identified by id_cols (e.g. ['neuron_id'] for FFN, ['head', 'neuron_id'] for attn),
    then averages across examples.

    Returns: Series indexed by layer (0..n_layers-1), values in [0, 1].
    """
    key = ["id", "layer"] + id_cols

    sets_a = (df_a.groupby(["id", "layer"])[id_cols]
              .apply(lambda g: set(map(tuple, g.values)))
              .rename("a"))
    sets_b = (df_b.groupby(["id", "layer"])[id_cols]
              .apply(lambda g: set(map(tuple, g.values)))
              .rename("b"))

    combined = sets_a.to_frame().join(sets_b, how="inner")
    combined["jaccard"] = combined.apply(
        lambda r: len(r["a"] & r["b"]) / len(r["a"] | r["b"])
        if r["a"] | r["b"] else np.nan,
        axis=1,
    )
    combined = combined.reset_index()
    return (combined.groupby("layer")["jaccard"].mean()
            .reindex(range(n_layers), fill_value=np.nan))


def plot_neuron_overlap(overlap_data, n_layers, path):
    """
    Pairwise Jaccard similarity of top-K neuron sets across prompt types, per layer.

    Two panels: FFN neurons (left) | Attn neurons (right).
    One line per pair: composition↔first-hop, composition↔second-hop, first-hop↔second-hop.

    Jaccard per (example, layer) = |intersection| / |union| of neuron identifiers.
    FFN: identified by neuron_id. Attn: identified by (head, neuron_id).
    Mean across examples shown with 95% CI band.

    High overlap → same neurons activate regardless of prompt type (shared circuit).
    Low overlap  → different neurons for different hops (separate circuits).

    overlap_data: dict prompt_type -> {"ffn_neurons": df, "attn_neurons": df}
                  (only prompt types with data are included)
    """
    pairs = [(a, b) for i, a in enumerate(overlap_data) for b in list(overlap_data)[i+1:]]
    if not pairs:
        print("  [SKIP] plot_neuron_overlap: need at least 2 prompt types")
        return

    pair_labels = {
        ("composition", "first-hop"):  "composition ↔ first-hop",
        ("composition", "second-hop"): "composition ↔ second-hop",
        ("first-hop",   "second-hop"): "first-hop ↔ second-hop",
    }

    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        x = np.arange(n_layers)

        for ax, neuron_key, id_cols, title in [
            (axes[0], "ffn_neurons",  ["neuron_id"],         "FFN neuron overlap"),
            (axes[1], "attn_neurons", ["head", "neuron_id"], "Attn neuron overlap"),
        ]:
            for pt_a, pt_b in pairs:
                df_a = overlap_data[pt_a].get(neuron_key)
                df_b = overlap_data[pt_b].get(neuron_key)
                if df_a is None or df_b is None:
                    continue
                label = pair_labels.get((pt_a, pt_b), f"{pt_a} ↔ {pt_b}")
                j = _jaccard_per_layer(df_a, df_b, id_cols, n_layers)
                ax.plot(x, j.values, "o-", markersize=3, label=label)

            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Mean Jaccard similarity")
            ax.set_xticks(range(0, n_layers, 2))
            ax.set_ylim(0, 1)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    layer_count = {
        "gpt2": 12,
        "meta-llama/Llama-2-7b-hf": 32,
        "meta-llama/Llama-2-13b-hf": 40,
        "meta-llama/Meta-Llama-3-8B": 32,
        "meta-llama/Meta-Llama-3-70B": 80,
    }
    head_count = {
        "gpt2": 12,
        "meta-llama/Llama-2-7b-hf": 32,
        "meta-llama/Llama-2-13b-hf": 40,
        "meta-llama/Meta-Llama-3-8B": 32,
        "meta-llama/Meta-Llama-3-70B": 64,
    }

    if args.prompt == "composition":
        attr_subdir = "neuron_attribution"
        prompt_suffix = ""
    elif args.prompt == "first-hop":
        attr_subdir = "neuron_attribution_first_hop"
        prompt_suffix = "_first_hop"
    elif args.prompt == "second-hop":
        attr_subdir = "neuron_attribution_second_hop"
        prompt_suffix = "_second_hop"

    for model in args.models:
        print(f"\nProcessing {model} (prompt={args.prompt})")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path   = f"{args.output_path}/{model}"
        n_layers = layer_count[model]
        n_heads  = head_count[model]

        lpi = {}
        for subset, label in zip(SUBSETS, SUBSET_LABELS):
            subset_dir = f"{dataset_path}/{subset}"
            try:
                lpi[label] = load_neuron_lpi(subset_dir, attr_subdir)
            except FileNotFoundError as e:
                print(f"  [SKIP] {label}: {e}")
                continue

        if not lpi:
            print(f"  [SKIP] No data loaded for {model}, skipping plots")
            continue

        loaded = list(lpi.keys())
        sublayer_data = {label: lpi[label]["sublayer"] for label in loaded}

        plot_sublayer_lpi_profile(sublayer_data, n_layers,
                                  f"{plots_path}/sublayer_lpi_profile{prompt_suffix}.pdf")
        print(f"Saved sublayer LPI profile -> {plots_path}/sublayer_lpi_profile{prompt_suffix}.pdf")

        plot_sublayer_lpi_bars(sublayer_data, n_layers,
                               f"{plots_path}/sublayer_lpi_bars{prompt_suffix}.pdf")
        print(f"Saved sublayer LPI bars    -> {plots_path}/sublayer_lpi_bars{prompt_suffix}.pdf")

        plot_head_lpi_heatmap(
            {label: lpi[label]["head"] for label in loaded},
            n_layers, n_heads,
            f"{plots_path}/head_lpi_heatmap{prompt_suffix}.pdf",
        )
        print(f"Saved head LPI heatmap     -> {plots_path}/head_lpi_heatmap{prompt_suffix}.pdf")

        plot_neuron_layer_distribution(
            {label: lpi[label]["ffn_neurons"]  for label in loaded},
            {label: lpi[label]["attn_neurons"] for label in loaded},
            n_layers,
            f"{plots_path}/neuron_layer_dist{prompt_suffix}.pdf",
        )
        print(f"Saved neuron layer dist    -> {plots_path}/neuron_layer_dist{prompt_suffix}.pdf")

        plot_query_layer_bars(
            {label: lpi[label]["ffn_query_layers"]  for label in loaded},
            {label: lpi[label]["attn_query_layers"] for label in loaded},
            n_layers,
            f"{plots_path}/query_layer_bars{prompt_suffix}.pdf",
        )
        print(f"Saved query layer bars     -> {plots_path}/query_layer_bars{prompt_suffix}.pdf")

        plot_query_value_neurons(
            {label: lpi[label]["attn_query_neurons"] for label in loaded},
            {label: lpi[label]["ffn_neurons"]        for label in loaded},
            n_layers,
            f"{plots_path}/query_value_neurons{prompt_suffix}.pdf",
        )
        print(f"Saved query value neurons  -> {plots_path}/query_value_neurons{prompt_suffix}.pdf")

        # ── Neuron overlap across prompt types (always attempted) ─────────────
        for subset, label in zip(SUBSETS, SUBSET_LABELS):
            subset_dir = f"{dataset_path}/{subset}"
            overlap_data = {}
            for pt, subdir in PROMPT_SUBDIR.items():
                try:
                    d = load_neuron_lpi(subset_dir, subdir)
                    overlap_data[pt] = {"ffn_neurons":  d["ffn_neurons"],
                                        "attn_neurons": d["attn_neurons"]}
                except FileNotFoundError:
                    pass
            if len(overlap_data) < 2:
                print(f"  [SKIP] neuron overlap {label}: need data for 2+ prompt types")
                continue
            out = f"{plots_path}/neuron_overlap_{label}.pdf"
            plot_neuron_overlap(overlap_data, n_layers, out)
            print(f"Saved neuron overlap ({label}) -> {out}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasets_dir")
    parser.add_argument("output_path")
    parser.add_argument("-m", "--models", nargs="+", required=True, choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B",
    ])
    parser.add_argument("--prompt", default="composition",
                        choices=["composition", "first-hop", "second-hop"],
                        help="Prompt type: selects which neuron_attribution_* directory to read")
    main(parser.parse_args())
