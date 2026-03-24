# %%
"""
Build layers.csv for each subset (if not cached) and print three summary tables.

Works for both pre-edit and post-edit subsets — pass any number of named
directory paths via --subsets.  Each directory must contain two_hop.csv and
the analysis outputs (sublayer_projection/, attention_knockout/,
entity_description/).

layers.csv is rebuilt from available source files if not present.
Delete _classified.csv files to force re-classification.

Usage:
    # Pre-edit (two subsets)
    python analysis/analyze_tables.py \\
        --subsets \\
            correct:datasets/meta-llama/Meta-Llama-3-8B/composition_correct \\
            incorrect:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect

    # Pre-edit + post-edit comparison
    python analysis/analyze_tables.py \\
        --subsets \\
            pre-edit:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect \\
            now_correct:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/now_correct \\
            still_incorrect:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/still_incorrect \\
            still_inconsistent:datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_alphaedit/still_inconsistent
"""

# %%

import ast
import sys
sys.path.insert(0, ".")
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


# %% Loaders

def load_layers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    for col in df.columns:
        if col not in ["id", "composition_correct"]:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) is not float else x)
    return df


def _parse_ranks(x):
    """Parse rank lists saved as Python lists '[1, 2]' or numpy arrays '[ 1  2]'."""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return [int(n) for n in str(x).strip("[]").split()]


def _classify_sublayer_projection(generations: pd.DataFrame, target: str):
    if generations[target].dtype == np.int64:
        return generations[target].apply(lambda x: x == 0)
    return generations[target].apply(lambda x: any(r == 0 for r in _parse_ranks(x)))


def load_sublayer_projections(dataset_dir: str) -> dict:
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


def build_layers_df(dataset_dir: str) -> pd.DataFrame:
    """
    Build layers DataFrame from available source files.
    Missing source files produce NaN columns (shown as N/A in tables).
    """
    tqdm.pandas()

    entity_description = load_entity_description_generations(dataset_dir)
    knockout           = load_attention_knockouts(dataset_dir)
    sublayer           = load_sublayer_projections(dataset_dir)

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


# %% Stats helpers

def to_min(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: min(x) if isinstance(x, list) and len(x) > 0 else np.nan)


def entity_decoding_stats(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    pct      = series.count() / len(df)
    mean_all = series.explode().mean()
    mean_min = series.apply(lambda x: min(x) if isinstance(x, list) else np.nan).mean()
    return {"pct": pct, "mean_all_layers": mean_all, "mean_min_layer": mean_min}


def info_prop_layer(df: pd.DataFrame) -> pd.Series:
    """
    Per-example mean layer where e2 emerges at t2, averaged across:
      (a) last_e1_attention_knockout_layers
      (b) attention_e2_projection_layers
      (c) last_e2_entity_layer
    Columns absent when the corresponding analysis was skipped are ignored.
    """
    cols = ["last_e1_attention_knockout_layers", "attention_e2_projection_layers", "last_e2_entity_layer"]
    parts = [to_min(df[c]) for c in cols if c in df.columns]
    if not parts:
        return pd.Series(dtype=float, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


ENTITY_COLUMNS = {
    "e2 from t1": "e1_e2_entity_layer",
    "e2 from t2": "last_e2_entity_layer",
    "e3 from t2": "last_e3_entity_layer",
}


# %% Tables

def print_tables(subsets: dict):
    w = max(len(name) for name in subsets) + 2

    print("Table 1: Entity decodability (PatchScopes)")
    print(f"{'Subset':<{w}} {'Entity':<12} {'% decoded':>10} {'mean layer (all)':>17} {'mean layer (min)':>17}")
    print("-" * (w + 60))
    for name, df in subsets.items():
        for label, col in ENTITY_COLUMNS.items():
            if col not in df.columns:
                print(f"{name:<{w}} {label:<12} {'N/A':>10}")
                continue
            s = entity_decoding_stats(df, col)
            print(f"{name:<{w}} {label:<12} {s['pct']:>9.1%}  {s['mean_all_layers']:>16.2f}  {s['mean_min_layer']:>16.2f}")
        print()

    print("\nTable 2: Info propagation layer (e2 emerges at t2)")
    print(f"{'Subset':<{w}} {'Cases':>7} {'Info prop layer':>16}")
    print("-" * (w + 26))
    for name, df in subsets.items():
        info = info_prop_layer(df)
        pct  = info.count() / len(df)
        print(f"{name:<{w}} {pct:>6.1%}  {info.mean():>16.2f}")

    print("\n\nTable 3: Mean layer where sublayer projection yields e3 as top prediction")
    print(f"{'Subset':<{w}} {'Sublayer':<10} {'Cases':>7} {'Mean layer (min)':>17}")
    print("-" * (w + 37))
    for name, df in subsets.items():
        for label, col in [("Attention", "attention_prediction_projection_layers"),
                           ("MLP",       "mlp_prediction_projection_layers")]:
            if col not in df.columns:
                print(f"{name:<{w}} {label:<10} {'N/A':>7}")
                continue
            s   = to_min(df[col])
            pct = s.count() / len(df)
            print(f"{name:<{w}} {label:<10} {pct:>6.1%}  {s.mean():>16.2f}")
        print()


# %% Main

def main(args):
    subsets = {}
    for entry in args.subsets:
        name, path = entry.split(":", 1)
        path = Path(path)
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
    parser.add_argument("--subsets", nargs="+", required=True,
                        metavar="NAME:PATH",
                        help="One or more name:path pairs, e.g. correct:datasets/.../composition_correct")
    main(parser.parse_args())
