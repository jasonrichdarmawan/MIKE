# %%
"""
Apply AlphaEdit to all incorrect examples, then run sublayer projection,
attention knockout, and patchscopes analyses on the fully-edited model.

Analyses are computed by method (not by subset): all subsets receive
sublayer projection results before attention knockout begins, and attention
knockout completes before the slower patchscopes step.

Subsets are derived from evaluated_two_hop.csv (post-edit evaluation):
    now_correct        — composition_correct == True after editing
    still_incorrect    — composition_correct == False, is_correct_second_hop == False
    still_inconsistent — composition_correct == False, is_correct_second_hop == True

Prerequisites:
    evaluated_two_hop.csv must exist in <output_dir>/.
    Generate it with evaluate_dataset.py run on the edited model.

Usage:
    python analysis/run_edited_analysis.py \\
        [--model  meta-llama/Meta-Llama-3-8B] \\
        [--hparams EasyEdit/hparams/AlphaEdit/llama3-8b.yaml] \\
        [--input  datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/two_hop.csv] \\
        [--output datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited] \\
        [--max-examples N] [--skip-patchscopes] [--device 0]
"""

import sys
sys.path.insert(0, "analysis")

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

from EasyEdit.easyeditor import AlphaEditHyperParams
from EasyEdit.easyeditor.util import nethook
from EasyEdit.easyeditor.util.alg_dict import ALG_DICT

from knockout_attention import generate_answer_with_attention_knockout
from patch_activations import activation_patching, create_target_prompt
from project_sublayer import project_attention
from utils import get_answers, load_model, load_tokenizer


# %% AlphaEdit helpers

def build_requests(dataset: pd.DataFrame) -> list:
    requests = []
    for _, row in dataset.iterrows():
        requests.append({"prompt": row["r1_template"], "subject": row["e1_label"], "target_new": row["e2_label"]})
        requests.append({"prompt": row["r2_template"], "subject": row["e2_label"], "target_new": row["e3_label"]})
    return requests


def save_checkpoint(model, hparams, ckpt_path: Path):
    weight_names = [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]
    weights = {k: nethook.get_parameter(model, k).detach().cpu() for k in weight_names}
    torch.save(weights, ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


def load_checkpoint(model, ckpt_path: Path, device: str):
    weights = torch.load(ckpt_path, map_location=device)
    with torch.no_grad():
        for k, v in weights.items():
            nethook.get_parameter(model, k)[...] = v.to(device)
    print(f"Loaded edited weights from {ckpt_path}")


# %% Subset splitting

def split_subsets(evaluated: pd.DataFrame) -> dict:
    """Split post-edit evaluated dataset into the three outcome subsets."""
    now_correct        = evaluated[evaluated["composition_correct"]]
    still_wrong        = evaluated[~evaluated["composition_correct"]]
    still_inconsistent = still_wrong[still_wrong["is_correct_second_hop"]]
    still_incorrect    = still_wrong[~still_wrong["is_correct_second_hop"]]
    return {
        "now_correct":        now_correct,
        "still_incorrect":    still_incorrect,
        "still_inconsistent": still_inconsistent,
    }


# %% Save helpers

def to_sublayer_df(ds) -> pd.DataFrame:
    """Convert HuggingFace dataset to DataFrame with proper Python list columns."""
    df = ds.to_pandas()
    for col in ("top_k_words", "top_k_probs", "e2_ranks", "e3_ranks"):
        if col in df.columns:
            df[col] = df[col].apply(list)
    return df


def save_by_subset(df: pd.DataFrame, subsets: dict, out_dir: Path,
                   subdir: str, filename: str, sort_by: list):
    """Filter df to each subset's IDs and save to out_dir/{subset}/{subdir}/{filename}."""
    for name, subset_df in subsets.items():
        filtered = df[df["id"].isin(subset_df["id"])]
        path = out_dir / name / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        filtered.sort_values(sort_by).reset_index(drop=True).to_csv(path, escapechar="\\")
        print(f"  [{name}] {len(filtered)} rows → {path}")


# %% Main

def main(args):
    out_dir   = Path(args.output)
    ckpt_path = out_dir / "editing_checkpoint.pt"
    device    = f"cuda:{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print(f"Loading model {args.model} ...")
    model     = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)

    # ── AlphaEdit ─────────────────────────────────────────────────────────────
    hparams    = AlphaEditHyperParams.from_hparams(args.hparams)
    apply_algo = ALG_DICT["AlphaEdit"]

    dataset = pd.read_csv(args.input, index_col=0)
    if args.max_examples:
        dataset = dataset.head(args.max_examples)
    dataset["e2_answers"] = dataset.apply(lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
    dataset["e3_answers"] = dataset.apply(lambda r: get_answers(r, "e3")["e3_answers"], axis=1)

    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device)
    else:
        requests = build_requests(dataset)
        print(f"Applying AlphaEdit ({len(requests)} facts across {len(dataset)} examples)...")
        apply_algo(model, tokenizer, requests, hparams, copy=False, return_orig_weights=False)
        save_checkpoint(model, hparams, ckpt_path)

    # ── Split into subsets ────────────────────────────────────────────────────
    evaluated_path = out_dir / "evaluated_two_hop.csv"
    if not evaluated_path.exists():
        raise FileNotFoundError(
            f"{evaluated_path} not found.\n"
            "Run evaluate_dataset.py on the edited model checkpoint first.")
    evaluated = pd.read_csv(evaluated_path, index_col=0)
    if args.max_examples:
        evaluated = evaluated.head(args.max_examples)

    subsets = split_subsets(evaluated)
    print("Subsets: " + ", ".join(f"{k}={len(v)}" for k, v in subsets.items()))

    for name, df in subsets.items():
        path = out_dir / name / "two_hop.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"Saved {path}")

    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    # ── Method 1: Sublayer projection — attention ─────────────────────────────
    print("\n=== Method 1/4: Sublayer projection (attention) ===")
    sublayer_attn = to_sublayer_df(hf_dataset.map(
        project_attention,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
        batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
    ))
    save_by_subset(sublayer_attn, subsets, out_dir,
                   "sublayer_projection", "attention_projections.csv", ["id", "layer"])

    # ── Method 2: Sublayer projection — MLP ───────────────────────────────────
    print("\n=== Method 2/4: Sublayer projection (MLP) ===")
    sublayer_mlp = to_sublayer_df(hf_dataset.map(
        project_attention,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
        batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
    ))
    save_by_subset(sublayer_mlp, subsets, out_dir,
                   "sublayer_projection", "mlp_projections.csv", ["id", "layer"])

    # ── Method 3: Attention knockout ──────────────────────────────────────────
    print("\n=== Method 3/4: Attention knockout ===")
    knockout = hf_dataset.map(
        generate_answer_with_attention_knockout,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                   "knockout_source": "last", "knockout_target": "e1", "k": 3,
                   "prompt": "composition"},
        batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
    ).to_pandas()
    save_by_subset(knockout, subsets, out_dir,
                   "attention_knockout", "last_e1_attention_knockout_k3_composition.csv",
                   ["id", "layer"])

    # ── Method 4: PatchScopes (optional, slow) ────────────────────────────────
    if not args.skip_patchscopes:
        print("\n=== Method 4/4: PatchScopes ===")
        for source in ["e1", "last"]:
            print(f"  source={source}")
            dataset_src = dataset.copy()
            dataset_src["target_prompt"] = create_target_prompt(dataset, source, "original")
            hf_src = Dataset.from_pandas(dataset_src, preserve_index=False)
            patch = hf_src.map(
                activation_patching,
                fn_kwargs={"model": model, "tokenizer": tokenizer,
                           "source": source, "target": "original", "default_decoding": False},
                batched=True, batch_size=args.batch_size,
                remove_columns=hf_src.column_names,
            ).to_pandas()
            save_by_subset(patch, subsets, out_dir,
                           "entity_description", f"{source}_entity_description_s0.csv",
                           ["id", "source_layer", "target_layer"])

    print(f"\nDone. Run: python analysis/analyze_edited.py")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",   default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--hparams", default="EasyEdit/hparams/AlphaEdit/llama3-8b.yaml")
    parser.add_argument("--input",
                        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/two_hop.csv")
    parser.add_argument("--output",
                        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-patchscopes", action="store_true",
                        help="Skip patchscopes (method 4) to save time")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    main(parser.parse_args())
