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
        "--model_name",
        type=str,
        help="Name of the model to analyze (e.g., gpt2, gpt2-medium, etc.)"
    )
    args = parser.parse_args()

    dataset_dir = Path("../datasets") / args.model_name

    inconsistent_df = load_activation_patching(dataset_dir, "inconsistent")
    incorrect_df = load_activation_patching(dataset_dir, "incorrect")

    for df, case_name in [(inconsistent_df, "inconsistent"), (incorrect_df, "incorrect")]:
        for source, target in [("e1", "original"), ("last", "original"), ("e1", "last")]:
            col_name = f"{source}_{target}_activation_patching_layer"
            backpatching_success = df[source][target][col_name].count() / len(df[source][target])
            print(f"Backpatching success for {case_name} cases, source: {source}, target: {target}: {backpatching_success:.4f}")

# %%
