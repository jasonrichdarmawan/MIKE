# %%
"""
Compute layers.csv for each post-edit subset and print the same three
summary tables as analyze_mechanism.py.

Post-edit subsets (under --edited dir):
    now_correct / still_incorrect / still_inconsistent

Each subset needs:
    attention_knockout/last_e1_attention_knockout_k3_composition.csv
    sublayer_projection/attention_projections.csv
    sublayer_projection/mlp_projections.csv
    entity_description/e1_entity_description_s0.csv      (optional)
    entity_description/last_entity_description_s0.csv    (optional)

layers.csv is rebuilt from available source files each run.
Delete _classified.csv files to force re-classification.

Usage:
    python analysis/analyze_edited.py
    python analysis/analyze_edited.py \\
        --base   datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect \\
        --edited datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited
"""

import ast
import sys
sys.path.insert(0, "analysis")

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from classify_results import (
    classify_generations,
    find_layers_by_classification,
    load_attention_knockouts,
    load_entity_description_generations,
)
from analyze_mechanism import (
    load_layers,
    entity_decoding_stats,
    info_prop_layer,
    to_min,
    ENTITY_COLUMNS,
)


# %% Numpy-safe rank parser
# (handles numpy-style rank strings from older runs; new runs produce Python lists)

def _parse_ranks(x):
    """Parse rank lists saved as either Python lists '[1, 2]' or numpy arrays '[ 1  2]'."""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return [int(n) for n in str(x).strip("[]").split()]


def _classify_sublayer_projection(generations: pd.DataFrame, target: str):
    if generations[target].dtype == np.int64:
        return generations[target].apply(lambda x: x == 0)
    return generations[target].apply(lambda x: any(r == 0 for r in _parse_ranks(x)))


def load_sublayer_projections_edited(dataset_dir: str) -> dict:
    layers = defaultdict(dict)
    for sublayer in ["attention", "mlp"]:
        path = f"{dataset_dir}/sublayer_projection/{sublayer}_projections.csv"
        if not Path(path).exists():
            continue
        generations = classify_generations(
            path, _classify_sublayer_projection, target="prediction_rank", use_cache=False)
        layers[sublayer]["prediction"] = find_layers_by_classification(
            generations, True, f"{sublayer}_prediction_projection_layers", False)
        generations = classify_generations(
            path, _classify_sublayer_projection, target="e2_ranks", use_cache=False)
        layers[sublayer]["e2"] = find_layers_by_classification(
            generations, True, f"{sublayer}_e2_projection_layers", False)
    return layers


# %% layers.csv builder

def build_layers_df(dataset_dir: str) -> pd.DataFrame:
    """
    Build layers DataFrame from available source files.
    Mirrors classify_results.main() — uses the same loaders.
    Missing source files produce NaN columns (shown as N/A in tables).
    """
    tqdm.pandas()

    entity_description = load_entity_description_generations(dataset_dir)
    knockout           = load_attention_knockouts(dataset_dir)
    sublayer           = load_sublayer_projections_edited(dataset_dir)

    two_hop = pd.read_csv(f"{dataset_dir}/two_hop.csv", index_col=0)
    df = two_hop[["id", "composition_correct"]]

    for source in ["e1", "last"]:
        for target in ["e2", "e3"]:
            layer_df = entity_description.get(source, {}).get(target)
            if layer_df is not None:
                df = df.merge(layer_df, on="id", how="left")

    for sub in ["attention", "mlp"]:
        for tgt in ["prediction", "e2"]:
            layer_df = sublayer.get(sub, {}).get(tgt)
            if layer_df is not None:
                df = df.merge(layer_df, on="id", how="left")

    ko_df = knockout.get("last", {}).get("e1")
    if ko_df is not None:
        df = df.merge(ko_df, on="id", how="left")

    return df


# %% Tables — same format as analyze_mechanism.py

def print_tables(subsets: dict):
    print("Table 1: Entity decodability (PatchScopes)")
    print(f"{'Subset':<28} {'Entity':<12} {'% decoded':>10} {'mean layer (all)':>17} {'mean layer (min)':>17}")
    print("-" * 88)
    for name, df in subsets.items():
        for label, col in ENTITY_COLUMNS.items():
            if col not in df.columns:
                print(f"{name:<28} {label:<12} {'N/A':>10}")
                continue
            s = entity_decoding_stats(df, col)
            print(f"{name:<28} {label:<12} {s['pct']:>9.1%}  {s['mean_all_layers']:>16.2f}  {s['mean_min_layer']:>16.2f}")
        print()

    print("\nTable 2: Info propagation layer (e2 emerges at t2)")
    print(f"{'Subset':<28} {'Cases':>7} {'Info prop layer':>16}")
    print("-" * 54)
    for name, df in subsets.items():
        info = info_prop_layer(df)
        pct  = info.count() / len(df)
        print(f"{name:<28} {pct:>6.1%}  {info.mean():>16.2f}")

    print("\n\nTable 3: Mean layer where sublayer projection yields e3 as top prediction")
    print(f"{'Subset':<28} {'Sublayer':<10} {'Cases':>7} {'Mean layer (min)':>17}")
    print("-" * 66)
    for name, df in subsets.items():
        for label, col in [("Attention", "attention_prediction_projection_layers"),
                           ("MLP",       "mlp_prediction_projection_layers")]:
            if col not in df.columns:
                print(f"{name:<28} {label:<10} {'N/A':>7}")
                continue
            s   = to_min(df[col])
            pct = s.count() / len(df)
            print(f"{name:<28} {label:<10} {pct:>6.1%}  {s.mean():>16.2f}")
        print()


# %% Main

def main(args):
    base_dir   = Path(args.base)
    edited_dir = Path(args.edited)

    subset_paths = {
        "pre-edit (incorrect)":     base_dir,
        "post: now_correct":        edited_dir / "now_correct",
        "post: still_incorrect":    edited_dir / "still_incorrect",
        "post: still_inconsistent": edited_dir / "still_inconsistent",
    }

    subsets = {}
    for name, path in subset_paths.items():
        if not path.exists():
            print(f"[SKIP] {name}: {path} not found")
            continue
        layers_csv = path / "layers.csv"
        if layers_csv.exists():
            print(f"Loading {name} from cached layers.csv ...")
            subsets[name] = load_layers(str(layers_csv))
        else:
            print(f"Building layers for {name} ...")
            df = build_layers_df(str(path))
            df.to_csv(layers_csv)
            subsets[name] = load_layers(str(layers_csv))

    print()
    print_tables(subsets)


# %%

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    parser.add_argument("--base",
        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect")
    parser.add_argument("--edited",
        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited")
    main(parser.parse_args())
