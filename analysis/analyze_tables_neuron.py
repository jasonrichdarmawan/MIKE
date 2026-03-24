# %%
"""
Tabular mechanistic analysis using neuron attribution results.

Answers four questions about the model (pre- or post-edit):

  Table 1 — Sublayer LPI (two-hop prompt)
    Which sublayer type (attention vs FFN) causally drives the final prediction,
    and at what layer does the peak contribution occur?

  Table 2 — Attribution chain mean layer
    Where in the network does each order of the backward attribution chain operate?
    Compares subsets to show where the causal pathway diverges.

  Table 3 — Neuron position distribution (order 0, two-hop)
    Which token positions (e1, r1, r2, last) do the top-K causal neurons respond to?

  Table 4/5 — Jaccard circuit overlap (two-hop vs second-hop / first-hop)
    Do examples reuse the same neurons for two-hop as for direct one-hop?

  Table 6/7/8 — Attribution flow map by prompt type
    Count-weighted mean layer broken down by (order, ntype, position),
    mapping to the 4-stage mechanism from Yu et al. (Back Attention, EMNLP 2025).

Usage:
    # Pre-edit
    python analysis/analyze_mechanism_neuron.py \\
        --subsets \\
            correct:datasets/meta-llama/Meta-Llama-3-8B/neuron_attribution/correct \\
            incorrect:datasets/meta-llama/Meta-Llama-3-8B/neuron_attribution/incorrect

    # Pre- and post-edit comparison
    python analysis/analyze_mechanism_neuron.py \\
        --subsets \\
            pre-edit:datasets/meta-llama/Meta-Llama-3-8B/neuron_attribution/incorrect \\
            now_correct:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited/neuron_attribution/now_correct \\
            still_incorrect:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited/neuron_attribution/still_incorrect
"""

# %%

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


PROMPT_TYPES = ["two-hop", "first-hop", "second-hop"]


def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


# %%
# ---------------------------------------------------------------------------
# Table 1: Sublayer LPI summary (two-hop prompt)
# ---------------------------------------------------------------------------

def table_sublayer_lpi(subsets: dict, prompt_type: str = "two-hop"):
    w = max(len(n) for n in subsets) + 2
    print(f"Table 1: Sublayer LPI summary ({prompt_type} prompt)")
    print(f"{'Subset':<{w}} {'Sublayer':<10} {'Mean LPI':>10} {'Peak layer':>12}")
    print("-" * (w + 36))
    for name, path in subsets.items():
        df = load_csv(f"{path}/{prompt_type}/sublayer_lpi.csv")
        if df is None:
            continue
        for sublayer in ["attention", "ffn"]:
            sub = df[df["sublayer"] == sublayer]
            mean_lpi   = sub["lpi"].mean()
            peak_layer = sub.groupby("layer")["lpi"].mean().idxmax()
            print(f"{name:<{w}} {sublayer:<10} {mean_lpi:>10.4f} {peak_layer:>12}")
        print()


# %%
# ---------------------------------------------------------------------------
# Table 2: Attribution chain mean layer by order
# ---------------------------------------------------------------------------

def table_chain_mean_layer(subsets: dict, prompt_type: str = "two-hop"):
    w = max(len(n) for n in subsets) + 2
    print(f"Table 2: Attribution chain mean layer ({prompt_type} prompt)")
    print(f"{'Subset':<{w}} {'Order':<8} {'NType':<12} {'Mean layer':>11} {'N neurons':>11}")
    print("-" * (w + 46))
    for name, path in subsets.items():
        df = load_csv(f"{path}/{prompt_type}/chain_mean_layer.csv")
        if df is None:
            continue
        for _, row in df.sort_values(["order", "ntype"], ascending=[False, True]).iterrows():
            ntype = row["ntype"].replace("\u2192", "->")
            print(f"{name:<{w}} {row['order']:<8} {ntype:<12} {row['mean_layer']:>11.2f} {int(row['n_neurons']):>11}")
        print()


# %%
# ---------------------------------------------------------------------------
# Table 3: Neuron position distribution (order 0, two-hop)
# ---------------------------------------------------------------------------

def table_position_dist(subsets: dict, prompt_type: str = "two-hop"):
    w = max(len(n) for n in subsets) + 2
    positions = ["r2", "r1", "e1", "last", "other"]
    print(f"Table 3: Neuron position distribution (order 0, {prompt_type} prompt)")
    print(f"{'Subset':<{w}} {'NType':<8} " + " ".join(f"{p:>8}" for p in positions))
    print("-" * (w + 8 + 9 * len(positions)))
    for name, path in subsets.items():
        df = load_csv(f"{path}/{prompt_type}/chain_position_dist.csv")
        if df is None:
            continue
        df0 = df[df["order"] == 0]
        for ntype in df0["ntype"].unique():
            sub   = df0[df0["ntype"] == ntype].set_index("position")["mean_count"]
            total = sub.sum()
            row_str = f"{name:<{w}} {ntype.replace(chr(8594), '->'):<8} "
            row_str += " ".join(f"{sub.get(p, 0) / total * 100:>7.1f}%" for p in positions)
            print(row_str)
        print()


# %%
# ---------------------------------------------------------------------------
# Tables 4/5: Jaccard circuit overlap
# ---------------------------------------------------------------------------

def table_jaccard_overlap(subsets: dict, pair: tuple[str, str], table_num: int):
    w = max(len(n) for n in subsets) + 2
    a, b = pair
    print(f"Table {table_num}: Jaccard circuit overlap — {a} vs {b} (order 0)")
    print(f"{'Subset':<{w}} {'Neuron class':<14} {'Mean Jaccard':>13}")
    print("-" * (w + 30))
    for name, path in subsets.items():
        df = load_csv(f"{path}/jaccard.csv")
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
            print(f"{name:<{w}} {nclass:<14} {grp['mean_jaccard'].mean():>13.4f}")
        print()


# %%
# ---------------------------------------------------------------------------
# Tables 6/7/8: Attribution flow map
# ---------------------------------------------------------------------------

def table_four_stage_mechanism(subsets: dict, prompt_type: str = "two-hop", table_num: int = 6):
    """
    Count-weighted mean layer for every (order, ntype, position) combination,
    ordered from deepest backward step to final prediction.
    """
    POSITION_ORDER = ["r2", "r1", "e1", "last", "other"]
    NTYPE_ORDER    = ["ffn", "attn", "attn\u2192ffn", "ffn\u2192attn"]

    w = max(len(n) for n in subsets) + 2
    dfs = {name: load_csv(f"{path}/{prompt_type}/chain_position_layer_dist.csv")
           for name, path in subsets.items()}

    combos = set()
    for df in dfs.values():
        if df is None:
            continue
        for _, row in df.iterrows():
            if row["mean_count"] > 0:
                combos.add((int(row["order"]), row["ntype"], row["position"]))
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
    for name in subsets:
        header += f" {name:>{max(12, len(name))}}"
    print(header)
    print("-" * (7 + 12 + 6 + (max(12, w) + 1) * len(subsets)))

    prev_order = None
    for order, ntype, position in combos:
        if prev_order is not None and order != prev_order:
            print()
        prev_order = order
        row_str = f"{order:<7} {ntype.replace(chr(8594), '->'):  <12} {position:<6}"
        for name, df in dfs.items():
            col_w = max(12, len(name))
            if df is None:
                row_str += f" {'N/A':>{col_w}}"
                continue
            sub = df[(df["order"] == order) & (df["ntype"] == ntype) & (df["position"] == position)]
            if sub.empty or sub["mean_count"].sum() == 0:
                row_str += f" {'N/A':>{col_w}}"
                continue
            mean_layer = (sub["layer"] * sub["mean_count"]).sum() / sub["mean_count"].sum()
            row_str += f" {mean_layer:>{col_w}.2f}"
        print(row_str)
    print()


# %%
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
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

    table_sublayer_lpi(subsets)
    print()
    table_chain_mean_layer(subsets)
    print()
    table_position_dist(subsets)
    print()
    table_jaccard_overlap(subsets, ("two-hop", "second-hop"), table_num=4)
    print()
    table_jaccard_overlap(subsets, ("two-hop", "first-hop"),  table_num=5)
    print()
    table_four_stage_mechanism(subsets, prompt_type="two-hop",    table_num=6)
    print()
    table_four_stage_mechanism(subsets, prompt_type="first-hop",  table_num=7)
    print()
    table_four_stage_mechanism(subsets, prompt_type="second-hop", table_num=8)


# %%

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", required=True,
                        metavar="NAME:PATH",
                        help="One or more name:path pairs where path is the neuron attribution "
                             "directory for that subset, e.g. "
                             "correct:datasets/meta-llama/Meta-Llama-3-8B/neuron_attribution/correct")
    main(parser.parse_args())
