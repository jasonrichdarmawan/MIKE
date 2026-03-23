"""
Tabular mechanistic analysis using neuron attribution results.

Answers four questions about the pre-edit model:

  Table 1 — Sublayer LPI (two-hop prompt)
    Which sublayer type (attention vs FFN) causally drives the final prediction,
    and at what layer does the peak contribution occur?

  Table 2 — Attribution chain mean layer
    Where in the network does each order of the backward attribution chain operate?
    Compares correct vs incorrect to show where the causal pathway diverges.

  Table 3 — Neuron position distribution (order 0, two-hop)
    Which token positions (e1, r1, r2, last) do the top-K causal neurons respond to?
    Differences between correct and incorrect reveal whether the failure is
    positional (wrong token attended to) or layer-level.

  Table 4 — Jaccard circuit overlap (two-hop vs second-hop)
    Do correct examples reuse the same neurons for two-hop as for direct second-hop?
    High overlap = compositional reuse; low overlap = broken/divergent circuit.

  Table 6 — 4-Stage Mechanism (order 0, two-hop)
    Count-weighted mean layer of top-K neurons broken down by (sublayer, position),
    mapping to Stages A–D from Yu et al. (Back Attention, EMNLP 2025):
      A = FFN @ e1 (low layers), B = Attn @ e1 (mid),
      C = FFN @ r1 (mid-high),   D = Attn @ last (high).

Usage:
    python analyze_mechanism_neuron.py -m meta-llama/Meta-Llama-3-8B
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


SUBSETS = ["correct", "incorrect", "inconsistent"]
PROMPT_TYPES = ["two-hop", "first-hop", "second-hop"]


def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Table 1: Sublayer LPI summary (two-hop prompt)
# ---------------------------------------------------------------------------

def table_sublayer_lpi(data_dir: str, subsets: list[str]):
    print("Table 1: Sublayer LPI summary (two-hop prompt)")
    print(f"{'Subset':<14} {'Sublayer':<10} {'Mean LPI':>10} {'Peak layer':>12}")
    print("-" * 50)
    for subset in subsets:
        path = f"{data_dir}/{subset}/two-hop/sublayer_lpi.csv"
        df = load_csv(path)
        if df is None:
            continue
        for sublayer in ["attention", "ffn"]:
            sub = df[df["sublayer"] == sublayer]
            mean_lpi = sub["lpi"].mean()
            peak_layer = sub.groupby("layer")["lpi"].mean().idxmax()
            print(f"{subset:<14} {sublayer:<10} {mean_lpi:>10.4f} {peak_layer:>12}")
        print()


# ---------------------------------------------------------------------------
# Table 2: Attribution chain mean layer by order
# ---------------------------------------------------------------------------

def table_chain_mean_layer(data_dir: str, subsets: list[str], prompt_type: str = "two-hop"):
    print(f"Table 2: Attribution chain mean layer ({prompt_type} prompt)")
    print(f"{'Subset':<14} {'Order':<8} {'NType':<12} {'Mean layer':>11} {'N neurons':>11}")
    print("-" * 60)
    for subset in subsets:
        path = f"{data_dir}/{subset}/{prompt_type}/chain_mean_layer.csv"
        df = load_csv(path)
        if df is None:
            continue
        for _, row in df.sort_values(["order", "ntype"], ascending=[False, True]).iterrows():
            ntype = row['ntype'].replace('\u2192', '->')
            print(f"{subset:<14} {row['order']:<8} {ntype:<12} {row['mean_layer']:>11.2f} {int(row['n_neurons']):>11}")
        print()


# ---------------------------------------------------------------------------
# Table 3: Neuron position distribution (order 0, two-hop)
# ---------------------------------------------------------------------------

def table_position_dist(data_dir: str, subsets: list[str]):
    print("Table 3: Neuron position distribution (order 0, two-hop prompt)")
    positions = ["r2", "r1", "e1", "last", "other"]
    header = f"{'Subset':<14} {'NType':<8} " + " ".join(f"{p:>8}" for p in positions)
    print(header)
    print("-" * (14 + 8 + 9 * len(positions)))
    for subset in subsets:
        path = f"{data_dir}/{subset}/two-hop/chain_position_dist.csv"
        df = load_csv(path)
        if df is None:
            continue
        df0 = df[df["order"] == 0]
        for ntype in df0["ntype"].unique():
            sub = df0[df0["ntype"] == ntype].set_index("position")["mean_count"]
            total = sub.sum()
            row_str = f"{subset:<14} {ntype.replace(chr(8594), '->'):<8} "
            row_str += " ".join(f"{sub.get(p, 0) / total * 100:>7.1f}%" for p in positions)
            print(row_str)
        print()


# ---------------------------------------------------------------------------
# Table 6: 4-Stage mechanism — count-weighted mean layer by (sublayer, position)
# ---------------------------------------------------------------------------

def table_four_stage_mechanism(data_dir: str, subsets: list[str], prompt_type: str = "two-hop", table_num: int = 6):
    """
    Full attribution flow map: count-weighted mean layer for every non-empty
    (order, ntype, position) combination, ordered from deepest backward step
    (order -2) to final prediction (order 0).

    Traces the 4-stage mechanism from Yu et al. (Back Attention, EMNLP 2025):
      FFN enrichment at e1 (order -2), r1/r2 (order -1) → back attention (order -1/-0) →
      final prediction (order 0).
    """
    POSITION_ORDER = ["r2", "r1", "e1", "last", "other"]
    NTYPE_ORDER    = ["ffn", "attn", "attn\u2192ffn", "ffn\u2192attn"]

    dfs = {}
    for subset in subsets:
        path = f"{data_dir}/{subset}/{prompt_type}/chain_position_layer_dist.csv"
        dfs[subset] = load_csv(path)

    # Collect all (order, ntype, position) combinations with data in any subset
    combos = set()
    for df in dfs.values():
        if df is None:
            continue
        for _, row in df.iterrows():
            if row["mean_count"] > 0:
                combos.add((int(row["order"]), row["ntype"], row["position"]))
    # Filter out "other" positions and sort: order asc (most negative first), then ntype, then position
    combos = sorted(
        [(o, nt, p) for o, nt, p in combos if p != "other"],
        key=lambda x: (
            x[0],
            NTYPE_ORDER.index(x[1]) if x[1] in NTYPE_ORDER else 99,
            POSITION_ORDER.index(x[2]) if x[2] in POSITION_ORDER else 99,
        )
    )

    print(f"Table {table_num}: Attribution flow map — count-weighted mean layer ({prompt_type})")
    header = f"{'Order':<7} {'NType':<12} {'Pos':<6}"
    for subset in subsets:
        header += f" {subset:>12}"
    print(header)
    print("-" * (7 + 12 + 6 + 13 * len(subsets)))

    prev_order = None
    for order, ntype, position in combos:
        if prev_order is not None and order != prev_order:
            print()  # blank line between orders
        prev_order = order
        ntype_label = ntype.replace('\u2192', '->')
        row_str = f"{order:<7} {ntype_label:<12} {position:<6}"
        for subset in subsets:
            df = dfs[subset]
            if df is None:
                row_str += f" {'N/A':>12}"
                continue
            sub = df[(df["order"] == order) & (df["ntype"] == ntype) & (df["position"] == position)]
            if sub.empty or sub["mean_count"].sum() == 0:
                row_str += f" {'N/A':>12}"
                continue
            mean_layer = (sub["layer"] * sub["mean_count"]).sum() / sub["mean_count"].sum()
            row_str += f" {mean_layer:>12.2f}"
        print(row_str)
    print()


def table_jaccard_overlap(data_dir: str, subsets: list[str], pair: tuple[str, str], table_num: int):
    a, b = pair
    print(f"Table {table_num}: Jaccard circuit overlap - {a} vs {b} (order 0)")
    print(f"{'Subset':<14} {'Neuron class':<14} {'Mean Jaccard':>13}")
    print("-" * 45)
    for subset in subsets:
        path = f"{data_dir}/{subset}/jaccard.csv"
        df = load_csv(path)
        if df is None:
            continue
        mask = (
            df["pair"].str.contains(a) &
            df["pair"].str.contains(b) &
            (df["order_a"] == 0) &
            (df["order_b"] == 0)
        )
        sub = df[mask]
        if sub.empty:
            continue
        for nclass, grp in sub.groupby("neuron_class"):
            print(f"{subset:<14} {nclass:<14} {grp['mean_jaccard'].mean():>13.4f}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    data_dir = f"{args.datasets_dir}/{args.model}/neuron_attribution"
    available = [s for s in SUBSETS if Path(f"{data_dir}/{s}").exists()]
    if not available:
        print(f"No neuron attribution data found under {data_dir}")
        return

    table_sublayer_lpi(data_dir, available)
    print()
    table_chain_mean_layer(data_dir, available)
    print()
    table_position_dist(data_dir, available)
    print()
    table_jaccard_overlap(data_dir, available, ("two-hop", "second-hop"), table_num=4)
    print()
    table_jaccard_overlap(data_dir, available, ("two-hop", "first-hop"), table_num=5)
    print()
    table_four_stage_mechanism(data_dir, available, prompt_type="two-hop",    table_num=6)
    print()
    table_four_stage_mechanism(data_dir, available, prompt_type="first-hop",  table_num=7)
    print()
    table_four_stage_mechanism(data_dir, available, prompt_type="second-hop", table_num=8)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasets_dir", nargs="?", default="datasets")
    parser.add_argument("-m", "--model", default="meta-llama/Meta-Llama-3-8B",
                        choices=["gpt2",
                                 "meta-llama/Meta-Llama-3-8B",
                                 "meta-llama/Llama-2-7b-hf",
                                 "meta-llama/Llama-2-13b-hf",
                                 "meta-llama/Meta-Llama-3-70B"])
    main(parser.parse_args())
