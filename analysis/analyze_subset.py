# %%

from pathlib import Path
import argparse

from classify_results import load_activation_patching

# %%
"""
Example:
- backpatch_t1: source == "e1", target == "original"
- backpatch_t2: source == "last", target == "original"
- t1_to_t2: source == "e1", target == "last"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        help="Path to the data directory"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    inconsistent_df = load_activation_patching(data_dir / "composition_inconsistent")
    incorrect_df = load_activation_patching(data_dir / "composition_incorrect")

    for df, case_name in [(inconsistent_df, "inconsistent"), (incorrect_df, "incorrect")]:
        for source, target in [("e1", "original"), ("last", "original"), ("e1", "last")]:
            col_name = f"{source}_{target}_activation_patching_layer"
            backpatching_success = df[source][target][col_name].count() / len(df[source][target])
            print(f"Backpatching success for {case_name} cases, source: {source}, target: {target}: {backpatching_success:.4f}")

# %%
