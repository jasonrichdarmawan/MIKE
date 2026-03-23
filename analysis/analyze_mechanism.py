import ast
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def load_layers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    for col in df.columns:
        if col not in ["id", "composition_correct"]:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) is not float else x)
    return df


def to_min(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: min(x) if isinstance(x, list) and len(x) > 0 else np.nan)


def entity_decoding_stats(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    pct = series.count() / len(df)
    mean_all = series.explode().mean()
    mean_min = series.apply(lambda x: min(x) if isinstance(x, list) else np.nan).mean()
    return {"pct": pct, "mean_all_layers": mean_all, "mean_min_layer": mean_min}


def info_prop_layer(df: pd.DataFrame) -> pd.Series:
    """
    Per-example mean layer where e2 emerges at t2 / attention from t2 to t1 is critical,
    averaged across three methods:
      (a) last_e1_attention_knockout_layers  — ablating attention from t2 to t1
      (b) attention_e2_projection_layers     — vocabulary projection of attention output
      (c) last_e2_entity_layer               — PatchScopes with indirect e2 representation
    """
    a = to_min(df["last_e1_attention_knockout_layers"])
    b = to_min(df["attention_e2_projection_layers"])
    c = to_min(df["last_e2_entity_layer"])
    return pd.concat([a, b, c], axis=1).mean(axis=1, skipna=True)


ENTITY_COLUMNS = {
    "e2 from t1": "e1_e2_entity_layer",
    "e2 from t2": "last_e2_entity_layer",
    "e3 from t2": "last_e3_entity_layer",
}


def main(args):
    subsets = {}
    for name, path in [("correct", args.correct), ("incorrect", args.incorrect)]:
        if not Path(path).exists():
            print(f"Skipping {name}: {path} not found")
            continue
        subsets[name] = load_layers(path)

    # --- Table 1: Entity decoding via PatchScopes ---
    print("Table 1: Entity decodability (PatchScopes)")
    print(f"{'Subset':<12} {'Entity':<12} {'% decoded':>10} {'mean layer (all)':>17} {'mean layer (min)':>17}")
    print("-" * 72)
    for name, df in subsets.items():
        for label, col in ENTITY_COLUMNS.items():
            if col not in df.columns:
                print(f"{name:<12} {label:<12} {'N/A':>10}")
                continue
            s = entity_decoding_stats(df, col)
            print(f"{name:<12} {label:<12} {s['pct']:>9.1%}  {s['mean_all_layers']:>16.2f}  {s['mean_min_layer']:>16.2f}")
        print()

    # --- Table 2: Info propagation layer (averaged across 3 methods) ---
    print("\nTable 2: Info propagation layer (e2 emerges at t2)")
    print(f"{'Subset':<12} {'Cases':>7} {'Info prop layer':>16}")
    print("-" * 38)
    for name, df in subsets.items():
        info = info_prop_layer(df)
        pct = info.count() / len(df)
        print(f"{name:<12} {pct:>6.1%}  {info.mean():>16.2f}")

    # --- Table 3: Vocabulary projection of sublayer output for e3 ---
    print("\nTable 3: Mean layer where sublayer projection yields e3 as top prediction")
    print(f"{'Subset':<12} {'Sublayer':<10} {'Cases':>7} {'Mean layer (min)':>17}")
    print("-" * 50)
    for name, df in subsets.items():
        for label, col in [("Attention", "attention_prediction_projection_layers"),
                           ("MLP",       "mlp_prediction_projection_layers")]:
            s = to_min(df[col])
            pct = s.count() / len(df)
            print(f"{name:<12} {label:<10} {pct:>6.1%}  {s.mean():>16.2f}")
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--correct", default="datasets/meta-llama/Meta-Llama-3-8B/composition_correct/layers.csv")
    parser.add_argument("--incorrect", default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/layers.csv")
    main(parser.parse_args())
