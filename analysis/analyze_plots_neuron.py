"""
Analyze neuron-level attribution results (LPI and residual stream decomposition) across subsets.

Two types of scores are used throughout:

  LPI (Log Probability Increase):
    How much a sublayer/head/neuron increases log P(predicted token) at the last position.
    Unit: nats (direct contribution to final prediction).
    Source: neuron_attribution.py -> sublayer_lpi.csv, head_lpi.csv.

  LPI-weighted residual contribution (query score):
    For a prior FFN neuron activating a top-K value neuron:
        score = (inner_product(neuron_output, V_key) / total_residual_alignment) * value_neuron_LPI
    Unit: fraction of residual stream alignment with the V-key vector, weighted by the
    downstream value neuron's LPI. Captures indirect upstream influence, not direct LPI.
    Source: neuron_attribution.py -> ffn_query_layers.csv, attn_query_layers.csv.

Produces:
  sublayer_lpi_profile.pdf    -- mean LPI per layer as lines (correct/incorrect/inconsistent)
  head_lpi_heatmap.pdf        -- mean head LPI heatmap (layer x head) per subset
  neuron_layer_dist.pdf       -- top-K FFN and attn neuron counts per layer, one line per subset
  query_layer_bars.pdf        -- residual stream decomposition: LPI-weighted contribution of
                                 prior layers to top-K value neurons
  chain_layer_profile.pdf     -- mean backward-chain score per layer at each order
  chain_mean_layer.pdf        -- score-weighted mean layer vs chain order per (prompt_type, subset)
  neuron_overlap_order{N}.pdf -- pairwise Jaccard similarity across prompt types at order N
  cross_order_{onehop}_{subset}.pdf  -- cross-order Jaccard heatmap: one-hop vs two-hop

Usage:
    # Pre-edit
    python analysis/analyze_plots_neuron.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --output-dir plots/meta-llama/Meta-Llama-3-8B \\
        --subsets \\
            correct:datasets/meta-llama/Meta-Llama-3-8B/correct/neuron_attribution \\
            incorrect:datasets/meta-llama/Meta-Llama-3-8B/incorrect/neuron_attribution

    # Pre- and post-edit comparison
    python analysis/analyze_plots_neuron.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --output-dir plots/meta-llama/Meta-Llama-3-8B/post_alphaedit \\
        --subsets \\
            pre-edit:datasets/meta-llama/Meta-Llama-3-8B/incorrect/neuron_attribution \\
            now_correct:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/now_correct/neuron_attribution \\
            still_incorrect:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/still_incorrect/neuron_attribution
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme()
sns.set_style("whitegrid")

SUBSET_LABELS = ["correct", "incorrect", "inconsistent"]
PROMPT_TYPES  = ["two-hop", "first-hop", "second-hop"]
PROMPT_TYPES  = ["two-hop", "first-hop", "second-hop"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_neuron_lpi(dataset_dir: str, attr_subdir: str = "two-hop"):
    """
    Load compact neuron attribution outputs produced by neuron_attribution.py.

    Returns dict with keys:
        sublayer, head, ffn_query_layers, attn_query_layers  — per-example DataFrames
        neuron_layer_dist  — mean_count per (order, ntype, layer)
        chain_profile      — mean_score per (order, ntype, layer)
        chain_mean_layer   — score-weighted mean layer per (order, ntype)
        chain_max_order    — max chain order found in chain_profile
    """
    path = Path(dataset_dir) / attr_subdir
    if not path.exists():
        raise FileNotFoundError(f"No neuron attribution directory found: {path}")

    def _optional(fname):
        p = path / fname
        return pd.read_csv(p) if p.exists() else None

    data = {
        "sublayer":             pd.read_csv(path / "sublayer_lpi.csv"),
        "head":                 pd.read_csv(path / "head_lpi.csv"),
        "ffn_query_layers":     _optional("ffn_query_layers.csv"),
        "attn_query_layers":    _optional("attn_query_layers.csv"),
        "neuron_layer_dist":    _optional("neuron_layer_dist.csv"),
        "chain_profile":        _optional("chain_profile.csv"),
        "chain_mean_layer":     _optional("chain_mean_layer.csv"),
        "chain_position_dist":       _optional("chain_position_dist.csv"),
        "chain_position_layer_dist": _optional("chain_position_layer_dist.csv"),
    }

    # Infer max chain order from chain_profile
    cp = data["chain_profile"]
    if cp is not None and len(cp) > 0:
        data["chain_max_order"] = int(cp["order"].min())  # most negative = deepest
    else:
        data["chain_max_order"] = 0

    return data


def load_overlap_position(subset_dir: str):
    """
    Load overlap_position.csv for a subset.

    Columns: pair, neuron_class, ntype_a, order_a, ntype_b, order_b,
             side ("a" or "b"), position, mean_count, n
    side="a" → position of overlapping neuron in pt_a; "b" → in pt_b.
    """
    p = Path(subset_dir) / "overlap_position.csv"
    return pd.read_csv(p) if p.exists() else None


def load_overlap_position_by_layer(subset_dir: str):
    """
    Load overlap_position_by_layer.csv for a subset.

    Columns: pair, neuron_class, ntype_a, order_a, ntype_b, order_b,
             layer, side ("a" or "b"), position, mean_count, n
    mean_count = average number of overlapping neurons at (layer, position) per example.
    """
    p = Path(subset_dir) / "overlap_position_by_layer.csv"
    return pd.read_csv(p) if p.exists() else None


def load_jaccard(subset_dir: str):
    """
    Load pre-computed Jaccard similarity CSV for a subset.

    Columns: pair, neuron_class, ntype_a, order_a, ntype_b, order_b, layer, mean_jaccard, n
    neuron_class is "ffn" or "attn" (groups same-class neurons across chain ntypes).
    Same-order same-ntype rows: used for value/query neuron overlap plots.
    Cross-order or cross-ntype same-class rows: used for chain-hop heatmaps.
    """
    p = Path(subset_dir) / "jaccard.csv"
    return pd.read_csv(p) if p.exists() else None


# ── Plot 1: sublayer LPI line profile (subset comparison) ────────────────────

def plot_sublayer_lpi_profile(lpi_data, n_layers: int, path: str):
    """
    Two-panel line plot (attention | FFN), mean LPI per layer with 95% CI,
    one line per subset.
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


# ── Plot 2: head LPI heatmap ─────────────────────────────────────────────────

def plot_head_lpi_heatmap(lpi_data, n_layers: int, n_heads: int, path: str):
    """
    Mean head LPI heatmap (layer x head) per subset, shared colour scale.
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


# ── Plot 4: neuron layer distribution ────────────────────────────────────────

def plot_neuron_layer_distribution(lpi_data, n_layers: int, path: str):
    """
    Average number of top-K neurons per layer per example, from compact
    neuron_layer_dist.csv (order=0 rows).

    Two panels: FFN neurons (left) | Attn neurons (right).
    """
    with sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        for ax, ntype, title in [
            (axes[0], "ffn",  "FFN value neurons"),
            (axes[1], "attn", "Attn value neurons"),
        ]:
            for label, lpi in lpi_data.items():
                dist = lpi.get("neuron_layer_dist")
                if dist is None or len(dist) == 0:
                    continue
                sub = dist[(dist["order"] == 0) & (dist["ntype"] == ntype)]
                counts = (
                    sub.set_index("layer")["mean_count"]
                    .reindex(range(n_layers), fill_value=0)
                )
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


# ── Plot 5: query layer bars ──────────────────────────────────────────────────

def plot_query_layer_bars(ffn_query_data, attn_query_data, n_layers, path):
    """
    Two-panel line plot: FFN query (left) | attn query (right).
    Mean LPI-weighted residual contribution per layer.
    """
    available_ffn  = {k: v for k, v in ffn_query_data.items()  if v is not None}
    available_attn = {k: v for k, v in attn_query_data.items() if v is not None}
    if not available_ffn and not available_attn:
        print("  [SKIP] plot_query_layer_bars: no query layer data available")
        return

    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
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


# ── Plot 6: neuron overlap across prompt types ────────────────────────────────

def plot_neuron_overlap(jaccard_df, n_layers, path, order=0):
    """
    Pairwise Jaccard similarity of top-K neuron sets across prompt types, per layer.

    Uses pre-computed jaccard.csv (same-order same-ntype rows: order_a == order_b == order,
    ntype_a == ntype_b). Filters by neuron_class ("ffn" | "attn") for two panels.
    Two panels: FFN neurons (left) | Attn neurons (right).
    """
    if jaccard_df is None or len(jaccard_df) == 0:
        print(f"  [SKIP] plot_neuron_overlap(order={order}): no jaccard data")
        return False

    sub = jaccard_df[
        (jaccard_df["order_a"] == order) &
        (jaccard_df["order_b"] == order) &
        (jaccard_df["ntype_a"] == jaccard_df["ntype_b"])
    ]
    if len(sub) == 0:
        print(f"  [SKIP] plot_neuron_overlap(order={order}): no data at this order")
        return False

    pair_labels = {
        "two-hop__first-hop":  "two-hop ↔ first-hop",
        "two-hop__second-hop": "two-hop ↔ second-hop",
        "first-hop__second-hop": "first-hop ↔ second-hop",
    }

    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        x = np.arange(n_layers)
        title_prefix = "Value neuron" if order == 0 else f"Order-{order} query neuron"

        for ax, nc, title in [
            (axes[0], "ffn",  f"{title_prefix} overlap (FFN)"),
            (axes[1], "attn", f"{title_prefix} overlap (Attn)"),
        ]:
            sub_nt = sub[sub["neuron_class"] == nc]
            for pair in sub_nt["pair"].unique():
                label = pair_labels.get(pair, pair)
                by_layer = (
                    sub_nt[sub_nt["pair"] == pair]
                    .groupby("layer")["mean_jaccard"].mean()
                    .reindex(range(n_layers), fill_value=np.nan)
                )
                ax.plot(x, by_layer.values, "o-", markersize=3, label=label)

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
    return True


# ── Plot 7: cross-order Jaccard heatmap ───────────────────────────────────────

def plot_cross_order_overlap(jaccard_df, n_layers, path, pair_key,
                             one_hop_label, two_hop_label="two-hop", vmax=None):
    """
    Cross-order Jaccard heatmap: one-hop chain orders (rows) vs two-hop chain orders (cols).

    Uses pre-computed jaccard.csv filtered by pair_key and neuron_class.
    Each cell = mean Jaccard similarity of the corresponding (order_a, order_b) pair,
    averaged over all layers. Supports cross-ntype same-class comparisons (e.g. first-hop
    order-0 "ffn" vs two-hop order-1 "ffn→attn"), enabling the +1-order-shift hypothesis.

    The three comparisons captured (for first-hop vs two-hop, FFN panel):
      [row=0, col=1] first-hop order-0 "ffn"     ↔ two-hop order-1 "ffn→attn"
      [row=1, col=2] first-hop order-1 "ffn→attn" ↔ two-hop order-2 "ffn→attn"
    And for Attn panel:
      [row=0, col=1] first-hop order-0 "attn"    ↔ two-hop order-1 "attn→ffn"

    Two panels: FFN neurons (left) | Attn neurons (right).
    pair_key: e.g. "two-hop__first-hop"
    """
    if jaccard_df is None or len(jaccard_df) == 0:
        print(f"  [SKIP] plot_cross_order_overlap({pair_key}): no jaccard data")
        return

    sub = jaccard_df[jaccard_df["pair"] == pair_key]
    if len(sub) == 0:
        print(f"  [SKIP] plot_cross_order_overlap({pair_key}): pair not found in jaccard.csv")
        return

    # pair_key = "pt_a__pt_b"; order_a corresponds to pt_a, order_b to pt_b
    pt_a, pt_b = pair_key.split("__")
    # rows = one_hop orders, cols = two_hop orders
    if pt_a == two_hop_label:
        row_col = "order_b"   # one_hop is pt_b
        col_col = "order_a"   # two_hop is pt_a
    else:
        row_col = "order_a"   # one_hop is pt_a
        col_col = "order_b"   # two_hop is pt_b

    row_orders = sorted(sub[row_col].unique())
    col_orders = sorted(sub[col_col].unique())

    with sns.plotting_context("paper", font_scale=1.2):
        fig, axes = plt.subplots(1, 2,
                                 figsize=(max(6, len(col_orders) * 1.5 + 2),
                                          max(4, len(row_orders) * 1.5 + 1)))

        for ax, nc, title in [
            (axes[0], "ffn",  "FFN neurons"),
            (axes[1], "attn", "Attn neurons"),
        ]:
            sub_nt = sub[sub["neuron_class"] == nc]
            matrix = np.full((len(row_orders), len(col_orders)), np.nan)

            for i, ro in enumerate(row_orders):
                for j, co in enumerate(col_orders):
                    cell = sub_nt[(sub_nt[row_col] == ro) & (sub_nt[col_col] == co)]
                    if len(cell) > 0:
                        # Average over layers (and over any ntype_a/ntype_b variants)
                        matrix[i, j] = cell["mean_jaccard"].mean()

            cell_max = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
            _vmax = vmax if vmax is not None else cell_max
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=_vmax,
                           origin="lower")
            ax.set_title(title)
            ax.set_xlabel(f"{two_hop_label} order")
            ax.set_ylabel(f"{one_hop_label} order" if ax == axes[0] else "")
            ax.set_xticks(range(len(col_orders)))
            ax.set_xticklabels(col_orders)
            ax.set_yticks(range(len(row_orders)))
            ax.set_yticklabels(row_orders)
            plt.colorbar(im, ax=ax, label="Mean Jaccard")

        fig.suptitle(f"Chain order overlap: {one_hop_label} vs {two_hop_label}")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 8: chained backward attribution layer profile ────────────────────────

def plot_chain_layer_profile(lpi_data, n_layers, path):
    """
    Grid of subplots: mean backward-chain score per layer at each order.

    Panels are determined by which (order, ntype) combinations have data.
    Expected layout for the fixed 2-order chain:
      order 1: "ffn→attn" (FFN at entity pos) | "attn→ffn" (attn activating last-pos FFN)
      order 2: "ffn→attn" (FFN activating order-1 attn)  [no "attn→ffn" at order 2]
    Uses pre-aggregated chain_profile.csv.
    """
    # Collect all (order, ntype) pairs present in any subset's data
    panels = set()
    for lpi in lpi_data.values():
        profile = lpi.get("chain_profile")
        if profile is None or len(profile) == 0:
            continue
        for _, row in profile[profile["order"] < 0].iterrows():
            panels.add((int(row["order"]), row["ntype"]))

    if not panels:
        print("  [SKIP] plot_chain_layer_profile: no chain data available")
        return

    # Stable order: sort by order descending (0, -1, -2) then ntype
    ntype_rank = {"ffn→attn": 0, "attn→ffn": 1}
    panels = sorted(panels, key=lambda x: (-x[0], ntype_rank.get(x[1], 2)))

    n_panels = len(panels)
    with sns.plotting_context("paper", font_scale=1.2):
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(6 * n_panels, 4),
                                 sharey=True)
        if n_panels == 1:
            axes = [axes]

        for ax, (order, ntype) in zip(axes, panels):
            for label, lpi in lpi_data.items():
                profile = lpi.get("chain_profile")
                if profile is None or len(profile) == 0:
                    continue
                sub = profile[(profile["order"] == order) & (profile["ntype"] == ntype)]
                if len(sub) == 0:
                    continue
                by_layer = (
                    sub.set_index("layer")["mean_score"]
                    .reindex(range(n_layers), fill_value=0)
                )
                ax.plot(range(n_layers), by_layer.values,
                        "o-", markersize=2, label=label)

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Layer")
            ax.set_xticks(range(0, n_layers, 2))
            ax.set_ylabel("Mean score")
            type_label = "FFN neurons" if ntype == "ffn→attn" else "Attn neurons"
            ax.set_title(f"Order {order} — {type_label}")
            ax.legend(loc="upper right", fontsize=7)

        fig.suptitle("Backward attribution chain by layer and order", y=1.02)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 9: chain mean layer ──────────────────────────────────────────────────

def plot_chain_mean_layer(summary_records, path):
    """
    Line plot of score-weighted mean layer vs chain order.
    One line per (prompt_type, subset) combination, two panels (FFN | Attn).

    summary_records: list of dicts with keys
        prompt_type, subset, order, ntype, mean_layer, n_neurons
    """
    df = pd.DataFrame(summary_records)
    if df.empty:
        print("  [SKIP] plot_chain_mean_layer: no data")
        return

    with sns.plotting_context("paper", font_scale=1.3):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

        for ax, ntype_prefix, title in [
            (axes[0], "ffn",  "FFN neurons — mean layer per order"),
            (axes[1], "attn", "Attn neurons — mean layer per order"),
        ]:
            # Match "ffn" (order 0) and "ffn→attn" (order > 0); same for attn.
            sub = df[df["ntype"].str.startswith(ntype_prefix)]
            for (prompt_type, subset), grp in sub.groupby(["prompt_type", "subset"]):
                grp = grp.sort_values("order", ascending=False)
                ax.plot(grp["order"], grp["mean_layer"], "o-", markersize=4,
                        label=f"{prompt_type} / {subset}")

            ax.set_title(title)
            ax.set_xlabel("Chain order (0 = value neurons; −1 = direct query; −2 = ffn→attn only)")
            ax.set_ylabel("Score-weighted mean layer" if ax == axes[0] else "")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(loc="upper right", fontsize=7, ncol=2)

        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 10: overlap position distribution ────────────────────────────────────

def plot_overlap_position(overlap_pos_df, path, pair_key, subset_label,
                          two_hop_label="two-hop", ymax=None):
    """
    For each theoretically motivated cross-order comparison, show where overlapping
    neurons sit in the sequence (e1, r1, r2, last, other).

    Two panels: FFN neurons (left) | Attn neurons (right).
    Within each panel, one group of bars per (order_a, order_b) comparison,
    split by position type. Side "a" = position in pt_a, "b" = position in pt_b.

    Since pt_a is typically two-hop (the richer chain), side="a" shows where the
    shared neurons are in the two-hop prompt — the key diagnostic.
    """
    if overlap_pos_df is None or len(overlap_pos_df) == 0:
        print(f"  [SKIP] plot_overlap_position({pair_key}, {subset_label}): no data")
        return

    sub = overlap_pos_df[overlap_pos_df["pair"] == pair_key]
    if len(sub) == 0:
        print(f"  [SKIP] plot_overlap_position({pair_key}): pair not found")
        return

    # Focus on side="a" (two-hop side) for the main diagnostic
    sub_a = sub[sub["side"] == "a"]

    pos_types = ["r2", "r1", "e1", "last", "other"]
    pt_a, pt_b = pair_key.split("__")
    # rows = one-hop orders, cols = two-hop orders (same convention as heatmap)
    if pt_a == two_hop_label:
        row_col, col_col = "order_b", "order_a"
    else:
        row_col, col_col = "order_a", "order_b"

    colors = sns.color_palette("Set2", len(pos_types))

    with sns.plotting_context("paper", font_scale=1.2):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

        for ax, nc, title in [
            (axes[0], "ffn",  "FFN neurons — overlap position in two-hop"),
            (axes[1], "attn", "Attn neurons — overlap position in two-hop"),
        ]:
            sub_nc = sub_a[sub_a["neuron_class"] == nc]
            if len(sub_nc) == 0:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Each comparison is one (order_a, order_b) pair — label as "orda→ordb"
            comparisons = sorted(sub_nc[[col_col, row_col]].drop_duplicates().itertuples(index=False),
                                 key=lambda r: (r[0], r[1]))
            x = np.arange(len(comparisons))
            bar_w = 0.8 / len(pos_types)

            for pi, (pos, color) in enumerate(zip(pos_types, colors)):
                heights = []
                for cmp in comparisons:
                    cell = sub_nc[
                        (sub_nc[col_col] == cmp[0]) & (sub_nc[row_col] == cmp[1]) &
                        (sub_nc["position"] == pos)
                    ]
                    heights.append(cell["mean_count"].sum() if len(cell) > 0 else 0.0)
                offset = (pi - len(pos_types) / 2 + 0.5) * bar_w
                ax.bar(x + offset, heights, bar_w, label=pos, color=color, alpha=0.85)

            cmp_labels = [f"{c[0]}→{c[1]}" for c in comparisons]
            ax.set_title(title)
            ax.set_xlabel(f"two-hop order → {pt_b.replace('-hop','')} order")
            ax.set_ylabel("Mean overlapping neuron count" if ax == axes[0] else "")
            ax.set_xticks(x)
            ax.set_xticklabels(cmp_labels, rotation=15)
            ax.legend(title="position", fontsize=7, loc="upper right")
            if ymax is not None:
                ax.set_ylim(0, ymax)

        fig.suptitle(f"Position of overlapping neurons: {pair_key} ({subset_label})")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 11: chain position distribution ─────────────────────────────────────

def plot_chain_position_dist(lpi_data, path):
    """
    For each (order, ntype), show the mean neuron count broken down by sequence
    position type: "e1", "e2", "last", "other".

    One panel per (order, ntype), bars grouped by position_type, one color per subset.
    Helps test the +1-shift hypothesis: order-2 FFN neurons should concentrate at
    e2 positions (bridge entity) more than order-0/1 neurons.
    """
    # Collect all (order, ntype) panels present in any subset
    panels = set()
    for lpi in lpi_data.values():
        cpd = lpi.get("chain_position_dist")
        if cpd is not None and len(cpd) > 0:
            for _, row in cpd.iterrows():
                panels.add((int(row["order"]), row["ntype"]))

    if not panels:
        print("  [SKIP] plot_chain_position_dist: no position data available")
        return

    ntype_rank = {"ffn": 0, "attn": 1, "ffn→attn": 2, "attn→ffn": 3}
    panels = sorted(panels, key=lambda x: (x[0], ntype_rank.get(x[1], 99)))
    pos_types = ["r2", "r1", "e1", "last", "other"]
    subsets = list(lpi_data.keys())
    x = np.arange(len(pos_types))
    bar_w = 0.8 / max(len(subsets), 1)
    colors = sns.color_palette("tab10", len(subsets))

    n_panels = len(panels)
    with sns.plotting_context("paper", font_scale=1.2):
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for ax, (order, ntype) in zip(axes, panels):
            for si, (label, lpi) in enumerate(lpi_data.items()):
                cpd = lpi.get("chain_position_dist")
                if cpd is None or len(cpd) == 0:
                    continue
                sub = cpd[(cpd["order"] == order) & (cpd["ntype"] == ntype)]
                counts = (
                    sub.set_index("position")["mean_count"]
                    .reindex(pos_types, fill_value=0)
                )
                offset = (si - len(subsets) / 2 + 0.5) * bar_w
                ax.bar(x + offset, counts.values, bar_w,
                       label=label, color=colors[si], alpha=0.85)

            ax.set_title(f"Order {order} — {ntype}")
            ax.set_xlabel("Position type")
            ax.set_ylabel("Mean neuron count" if ax == axes[0] else "")
            ax.set_xticks(x)
            ax.set_xticklabels(pos_types)
            ax.legend(loc="upper right", fontsize=7)

        fig.suptitle("Neuron position distribution by chain order", y=1.02)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 12: overlap position × layer heatmap ────────────────────────────────

def plot_overlap_position_by_layer(overlap_pos_layer_df, n_layers, path, pair_key,
                                   subset_label, two_hop_label="two-hop", vmax=None):
    """
    Heatmap of overlapping neuron count by (layer, position) for each comparison.

    One figure per (pair_key, subset_label).
    Rows of the figure: neuron_class (FFN | Attn).
    Cols of the figure: unique (order_a, order_b) comparisons, sorted by order.
    Each cell: heatmap where x = position type, y = layer, colour = mean_count.
    Side "a" (two-hop side) is shown — it is the richer chain and the key diagnostic.
    """
    if overlap_pos_layer_df is None or len(overlap_pos_layer_df) == 0:
        print(f"  [SKIP] plot_overlap_position_by_layer({pair_key}, {subset_label}): no data")
        return

    sub = overlap_pos_layer_df[
        (overlap_pos_layer_df["pair"] == pair_key) &
        (overlap_pos_layer_df["side"] == "a")
    ]
    if len(sub) == 0:
        print(f"  [SKIP] plot_overlap_position_by_layer({pair_key}): pair not found")
        return

    # Unique (order_a, order_b) comparisons: same-order first (diagonal), then cross-order
    comparisons = sorted(
        sub[["order_a", "order_b"]].drop_duplicates().itertuples(index=False),
        key=lambda r: (r.order_a != r.order_b, r.order_a, r.order_b)
    )
    pos_types = ["r2", "r1", "e1", "last", "other"]
    ncs = ["ffn", "attn"]
    n_rows = len(ncs)
    n_cols = len(comparisons)

    vmax = vmax if vmax is not None else (sub["mean_count"].max() if len(sub) > 0 else 1.0)

    with sns.plotting_context("paper", font_scale=1.1):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(3, 1.4 * len(pos_types)) * n_cols,
                     max(3, 0.4 * n_layers) * n_rows),
            squeeze=False,
            constrained_layout=True,
        )

        for ri, nc in enumerate(ncs):
            for ci, cmp in enumerate(comparisons):
                ax = axes[ri][ci]
                cell = sub[
                    (sub["neuron_class"] == nc) &
                    (sub["order_a"] == cmp.order_a) &
                    (sub["order_b"] == cmp.order_b)
                ]
                # Build layer × position matrix via pivot
                matrix = np.zeros((n_layers, len(pos_types)))
                if len(cell) > 0:
                    piv = (
                        cell[cell["position"].isin(pos_types)]
                        .pivot_table(index="layer", columns="position",
                                     values="mean_count", aggfunc="sum", fill_value=0)
                        .reindex(index=range(n_layers), columns=pos_types, fill_value=0)
                    )
                    matrix = piv.values

                im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                               vmin=0, vmax=vmax, origin="lower")
                ax.set_xticks(range(len(pos_types)))
                ax.set_xticklabels(pos_types, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(0, n_layers, max(1, n_layers // 6)))
                ax.set_yticklabels(range(0, n_layers, max(1, n_layers // 6)), fontsize=8)

                title_nc = "FFN" if nc == "ffn" else "Attn"
                ax.set_title(f"{title_nc} | ord {cmp.order_a}↔{cmp.order_b}", fontsize=9)
                if ci == 0:
                    ax.set_ylabel("Layer")
                if ri == n_rows - 1:
                    ax.set_xlabel("Position")

        fig.colorbar(
            plt.cm.ScalarMappable(cmap="YlOrRd",
                                  norm=plt.Normalize(vmin=0, vmax=vmax)),
            ax=axes, orientation="vertical", label="Mean overlapping neuron count",
            fraction=0.02, pad=0.04,
        )
        fig.suptitle(
            f"Overlap position × layer: {pair_key} ({subset_label})\n"
            f"Two-hop side shown",
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ── Plot 13: attribution chain flow heatmap ──────────────────────────────────

def plot_four_stage_mechanism(lpi_data, n_layers, path):
    """
    Full attribution chain flow: position × layer heatmap for each chain step.

    Two parallel chains feeding the final prediction:
      Chain A: ffn→attn (order -2) → attn→ffn (order -1) → ffn (order 0, last)
      Chain B: ffn→attn (order -1) → attn (order 0, all positions)

    Columns = chain steps ordered deepest → shallowest. Order -2 is skipped if
    not present in data (first-hop / second-hop only have orders 0 and -1).
    Rows = correct / incorrect subsets.
    Each panel: y = position (r2/r1/e1/last), x = layer, colour = mean neuron count.
    Shared colour scale across all panels.
    """
    positions = ["r2", "r1", "e1", "last"]

    # All possible chain steps in display order (left = deepest backward)
    ALL_STEPS = [
        (-2, "ffn\u2192attn",  "Chain A\norder −2\nffn→attn"),
        (-1, "attn\u2192ffn",  "Chain A\norder −1\nattn→ffn"),
        (-1, "ffn\u2192attn",  "Chain B\norder −1\nffn→attn"),
        ( 0, "ffn",            "Chain A\norder 0\nffn"),
        ( 0, "attn",           "Chain B\norder 0\nattn"),
    ]

    row_labels = list(lpi_data.keys())

    # Check data availability
    available_dfs = {
        label: lpi_data[label]["chain_position_layer_dist"]
        for label in row_labels
        if lpi_data[label].get("chain_position_layer_dist") is not None
    }
    if not available_dfs:
        print("  [SKIP] plot_four_stage_mechanism: no chain_position_layer_dist data")
        return

    # Determine which steps have data (skip order -2 if absent)
    all_data = pd.concat(available_dfs.values(), ignore_index=True)
    steps = [
        (order, ntype, label)
        for order, ntype, label in ALL_STEPS
        if len(all_data[(all_data["order"] == order) & (all_data["ntype"] == ntype)]) > 0
    ]
    if not steps:
        print("  [SKIP] plot_four_stage_mechanism: no chain steps found in data")
        return

    # Shared vmax
    vmax = float(all_data["mean_count"].max()) if len(all_data) > 0 else 1.0

    n_rows = len(row_labels)
    n_cols = len(steps)
    with sns.plotting_context("paper", font_scale=1.0):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.8 * n_cols, 2.5 * n_rows),
            sharex=True, sharey=True,
            squeeze=False,
        )

        for ri, label in enumerate(row_labels):
            df = available_dfs.get(label)
            for ci, (order, ntype, col_label) in enumerate(steps):
                ax = axes[ri][ci]
                mat = np.zeros((len(positions), n_layers))
                if df is not None:
                    sub = df[(df["order"] == order) & (df["ntype"] == ntype)]
                    for _, row in sub.iterrows():
                        if row["position"] in positions and 0 <= int(row["layer"]) < n_layers:
                            pi = positions.index(row["position"])
                            mat[pi, int(row["layer"])] = row["mean_count"]

                im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                               vmin=0, vmax=vmax, origin="lower")
                ax.set_yticks(range(len(positions)))
                ax.set_yticklabels(positions if ci == 0 else [], fontsize=8)
                ax.set_xticks(range(0, n_layers, max(1, n_layers // 8)))
                ax.set_xticklabels(
                    range(0, n_layers, max(1, n_layers // 8)) if ri == n_rows - 1 else [],
                    fontsize=7,
                )
                if ri == 0:
                    ax.set_title(col_label, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=9)
                if ri == n_rows - 1:
                    ax.set_xlabel("Layer", fontsize=8)

        fig.colorbar(
            plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=vmax)),
            ax=axes, orientation="vertical", label="Mean neuron count",
            fraction=0.02, pad=0.04,
        )
        fig.suptitle("Attribution chain flow: position × layer", y=1.01)
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

    n_layers = layer_count[args.model]
    n_heads  = head_count[args.model]
    plots_path = args.output_dir
    prompt_suffix = "" if args.prompt == "two-hop" else f"_{args.prompt.replace('-', '_')}"

    # ── Parse subsets ─────────────────────────────────────────────────────────
    subsets = {}
    for entry in args.subsets:
        name, path = entry.split(":", 1)
        if not Path(path).exists():
            print(f"[SKIP] {name}: {path} not found")
            continue
        subsets[name] = path

    if not subsets:
        print("No valid subset paths found.")
        return

    # ── Per-subset LPI data (for the selected prompt type) ────────────────────
    lpi = {}
    for name, path in subsets.items():
        try:
            lpi[name] = load_neuron_lpi(path, args.prompt)
        except FileNotFoundError as e:
            print(f"  [SKIP] {name}: {e}")

    if not lpi:
        print("No neuron attribution data loaded.")
        return

    loaded = list(lpi.keys())

    # ── Per-prompt-type single-prompt plots ───────────────────────────────────
    plot_sublayer_lpi_profile(
        {name: lpi[name]["sublayer"] for name in loaded},
        n_layers,
        f"{plots_path}/sublayer_lpi_profile{prompt_suffix}.pdf",
    )
    print(f"Saved sublayer LPI profile -> {plots_path}/sublayer_lpi_profile{prompt_suffix}.pdf")

    plot_head_lpi_heatmap(
        {name: lpi[name]["head"] for name in loaded},
        n_layers, n_heads,
        f"{plots_path}/head_lpi_heatmap{prompt_suffix}.pdf",
    )
    print(f"Saved head LPI heatmap     -> {plots_path}/head_lpi_heatmap{prompt_suffix}.pdf")

    plot_neuron_layer_distribution(
        lpi, n_layers,
        f"{plots_path}/neuron_layer_dist{prompt_suffix}.pdf",
    )
    print(f"Saved neuron layer dist    -> {plots_path}/neuron_layer_dist{prompt_suffix}.pdf")

    plot_query_layer_bars(
        {name: lpi[name]["ffn_query_layers"]  for name in loaded},
        {name: lpi[name]["attn_query_layers"] for name in loaded},
        n_layers,
        f"{plots_path}/query_layer_bars{prompt_suffix}.pdf",
    )
    print(f"Saved query layer bars     -> {plots_path}/query_layer_bars{prompt_suffix}.pdf")

    plot_chain_layer_profile(
        lpi, n_layers,
        f"{plots_path}/chain_layer_profile{prompt_suffix}.pdf",
    )
    print(f"Saved chain layer profile  -> {plots_path}/chain_layer_profile{prompt_suffix}.pdf")

    plot_chain_position_dist(
        lpi,
        f"{plots_path}/chain_position_dist{prompt_suffix}.pdf",
    )
    print(f"Saved chain position dist  -> {plots_path}/chain_position_dist{prompt_suffix}.pdf")

    for pt in PROMPT_TYPES:
        lpi_pt = {}
        for name, path in subsets.items():
            try:
                lpi_pt[name] = load_neuron_lpi(path, pt)
            except FileNotFoundError:
                continue
        if lpi_pt:
            pt_suffix = f"_{pt.replace('-', '_')}"
            plot_four_stage_mechanism(
                lpi_pt, n_layers,
                f"{plots_path}/chain_flow{pt_suffix}.pdf",
            )
            print(f"Saved chain flow ({pt})  -> {plots_path}/chain_flow{pt_suffix}.pdf")

    # ── Cross-prompt plots (use pre-computed Jaccard CSVs) ────────────────────
    summary_records = []

    # Compute shared vmaxes across all subsets for comparable colour scales
    if args.heatmap_vmax is not None:
        heatmap_vmax = args.heatmap_vmax
    else:
        all_jaccards = []
        for path in subsets.values():
            _jdf = load_jaccard(path)
            if _jdf is not None and len(_jdf) > 0:
                all_jaccards.extend(_jdf["mean_jaccard"].dropna().tolist())
        heatmap_vmax = float(np.nanmax(all_jaccards)) if all_jaccards else 1.0
        print(f"  heatmap vmax (global): {heatmap_vmax:.3f}")

    all_pos_counts = []
    for path in subsets.values():
        _df = load_overlap_position(path)
        if _df is not None and len(_df) > 0:
            all_pos_counts.extend(_df["mean_count"].dropna().tolist())
    overlap_pos_ymax = float(np.nanmax(all_pos_counts)) if all_pos_counts else 1.0
    print(f"  overlap position ymax (global): {overlap_pos_ymax:.3f}")

    all_pos_layer_counts = []
    for path in subsets.values():
        _df = load_overlap_position_by_layer(path)
        if _df is not None and len(_df) > 0:
            all_pos_layer_counts.extend(_df["mean_count"].dropna().tolist())
    overlap_pos_layer_vmax = float(np.nanmax(all_pos_layer_counts)) if all_pos_layer_counts else 1.0
    print(f"  overlap pos×layer vmax (global): {overlap_pos_layer_vmax:.3f}")

    for name, path in subsets.items():
        jaccard_df           = load_jaccard(path)
        overlap_pos_df       = load_overlap_position(path)
        overlap_pos_layer_df = load_overlap_position_by_layer(path)

        # Same-order Jaccard overlap plots
        if jaccard_df is not None and len(jaccard_df) > 0:
            orders_in_jaccard = sorted(
                set(jaccard_df["order_a"].unique()) | set(jaccard_df["order_b"].unique())
            )
            for order in orders_in_jaccard:
                slug_order = str(order).replace("-", "neg")
                out = f"{plots_path}/neuron_overlap_order{slug_order}_{name}{prompt_suffix}.pdf"
                if plot_neuron_overlap(jaccard_df, n_layers, out, order=order):
                    print(f"Saved neuron overlap (order={order}, {name}) -> {out}")

        # Cross-order Jaccard heatmaps: each one-hop vs two-hop
        if jaccard_df is not None:
            pairs_present = set(jaccard_df["pair"].unique())
            for one_hop_pt in ("first-hop", "second-hop"):
                slug = one_hop_pt.replace("-", "_")
                pair_key_ab = f"two-hop__{one_hop_pt}"
                pair_key_ba = f"{one_hop_pt}__two-hop"
                if pair_key_ab in pairs_present:
                    pair_key = pair_key_ab
                elif pair_key_ba in pairs_present:
                    pair_key = pair_key_ba
                else:
                    print(f"  [SKIP] cross-order {name}/{one_hop_pt}: pair not in jaccard.csv")
                    continue

                out = f"{plots_path}/cross_order_{slug}_{name}{prompt_suffix}.pdf"
                plot_cross_order_overlap(jaccard_df, n_layers, out, pair_key,
                                         one_hop_label=one_hop_pt, vmax=heatmap_vmax)
                print(f"Saved cross-order overlap ({one_hop_pt}, {name}) -> {out}")

                out = f"{plots_path}/overlap_position_{slug}_{name}{prompt_suffix}.pdf"
                plot_overlap_position(overlap_pos_df, out, pair_key, name,
                                      ymax=overlap_pos_ymax)
                print(f"Saved overlap position   ({one_hop_pt}, {name}) -> {out}")

                out = f"{plots_path}/overlap_pos_layer_{slug}_{name}{prompt_suffix}.pdf"
                plot_overlap_position_by_layer(overlap_pos_layer_df, n_layers,
                                               out, pair_key, name,
                                               vmax=overlap_pos_layer_vmax)
                print(f"Saved overlap pos×layer  ({one_hop_pt}, {name}) -> {out}")

        # Chain mean layer summary
        for pt in PROMPT_TYPES:
            try:
                pt_data = load_neuron_lpi(path, pt)
            except FileNotFoundError:
                continue
            cml = pt_data.get("chain_mean_layer")
            if cml is None or len(cml) == 0:
                continue
            for _, row in cml.iterrows():
                summary_records.append({
                    "subset":      name,
                    "prompt_type": pt,
                    "order":       int(row["order"]),
                    "ntype":       row["ntype"],
                    "mean_layer":  float(row["mean_layer"]),
                    "n_neurons":   int(row["n_neurons"]),
                })

    if summary_records:
        csv_path = f"{plots_path}/chain_layer_summary.csv"
        pd.DataFrame(summary_records).to_csv(csv_path, index=False)
        print(f"Saved chain layer summary  -> {csv_path}")
        plot_chain_mean_layer(summary_records, f"{plots_path}/chain_mean_layer.pdf")
        print(f"Saved chain mean layer     -> {plots_path}/chain_mean_layer.pdf")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--subsets", nargs="+", required=True,
                        metavar="NAME:PATH",
                        help="One or more name:path pairs where path is the neuron attribution "
                             "subset directory, e.g. "
                             "correct:datasets/meta-llama/Meta-Llama-3-8B/correct/neuron_attribution")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write plot PDFs into")
    parser.add_argument("-m", "--model", required=True,
                        choices=["gpt2",
                                 "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
                                 "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"],
                        help="Model name (used for layer/head count metadata only)")
    parser.add_argument("--prompt", default="two-hop",
                        choices=["two-hop", "first-hop", "second-hop"],
                        help="Prompt type to use for per-subset plots")
    parser.add_argument("--heatmap-vmax", type=float, default=None,
                        help="Fixed colour-scale max for cross-order heatmaps "
                             "(default: auto from global max across all subsets)")
    main(parser.parse_args())
