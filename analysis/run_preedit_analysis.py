# %%
"""
Evaluate the base model, create pre-edit subsets, then run sublayer projection,
attention knockout, and patchscopes — all in one pass.

Pipeline:
  1. Evaluate base model on raw dataset (first/second hop, composition, shortcuts)
  2. Filter shortcuts and create correct / incorrect / inconsistent subsets
     with stratified sampling (mirrors create_subset.py)
  3. Run sublayer projection, attention knockout, and patchscopes per subset

Results are written to:
    <data-dir>/composition_{subset}/sublayer_projection/
    <data-dir>/composition_{subset}/attention_knockout/
    <data-dir>/composition_{subset}/entity_description/

Usage:
    python analysis/run_preedit_analysis.py \\
        [--model    meta-llama/Meta-Llama-3-8B] \\
        [--input    datasets/two_hop.csv] \\
        [--data-dir datasets/meta-llama/Meta-Llama-3-8B] \\
        [--max-examples N] [--skip-patchscopes] [--batch-size 32]
"""

# %%

import sys
sys.path.insert(0, "analysis")

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from evaluate_dataset import (
    eval_composition, eval_first_hop, eval_second_hop,
    eval_entity_shortcut, eval_relation_shortcut,
)
from generate_entity_description import generate_entity_description_with_patching
from knockout_attention import generate_answer_with_attention_knockout
from project_sublayer import project_attention
from run_alphaedit_analysis import to_sublayer_df
from utils import get_answers, load_model, load_tokenizer


# %%

TARGET_PROMPT = (
    "Syria: Syria is a country in the Middle East, "
    "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, "
    "Samsung: Samsung is a South Korean multinational corporation, "
    "x"
)


# %% Subset creation (mirrors create_subset.py)

def sample_group(group: pd.DataFrame, n: int) -> pd.DataFrame:
    return group.sample(n=min(len(group), n), random_state=42)


def create_subsets(evaluated: pd.DataFrame, data_dir: Path) -> dict:
    """
    Filter shortcuts, label is_correct_second_hop, stratified-sample, and save.

    Subsets:
      correct      — composition + both hops correct, no shortcuts (n=100 per type)
      incorrect    — both hops correct but composition fails in all contexts (n=50 per type)
      inconsistent — both hops correct but composition fails; bridge entity e2 is solvable
                     in some other context (n=30 per type)
    """
    df = evaluated[~evaluated["entity_shortcut_correct"] & ~evaluated["relation_shortcut_correct"]]
    df_correct   = df[df["composition_correct"]  & df["first_hop_correct"] & df["second_hop_correct"]]
    df_incorrect = df[~df["composition_correct"] & df["first_hop_correct"] & df["second_hop_correct"]]

    # Label inconsistent: e2 context (e2, r2, e3) appears in correct set
    correct_second_hop = set(
        (r["e2_label"], r["r2_type"], r["e3_label"])
        for r in tqdm(df_correct.to_dict("records"), desc="Building correct second-hop set")
    )
    df_incorrect = df_incorrect.copy()
    df_incorrect["is_correct_second_hop"] = df_incorrect.apply(
        lambda r: (r["e2_label"], r["r2_type"], r["e3_label"]) in correct_second_hop, axis=1
    )
    df_inconsistent = df_incorrect[df_incorrect["is_correct_second_hop"]]
    df_incorrect_only = df_incorrect[~df_incorrect["is_correct_second_hop"]]

    subsets = {
        "correct":      df_correct.groupby(["r1_type", "r2_type"], group_keys=False).apply(sample_group, n=100),
        "incorrect":    df_incorrect_only.groupby(["r1_type", "r2_type"], group_keys=False).apply(sample_group, n=50),
        "inconsistent": df_inconsistent.groupby(["r1_type", "r2_type"], group_keys=False).apply(sample_group, n=30),
    }

    for name, df_sub in subsets.items():
        out = data_dir / f"composition_{name}" / "two_hop.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df_sub.to_csv(out)
        print(f"Saved {name} subset ({len(df_sub)} examples) -> {out}")

    return subsets


# %% Main

def main(args):
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print(f"Loading model {args.model} ...")
    model     = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"

    # ── Evaluate base model ───────────────────────────────────────────────────
    evaluated_path = data_dir / "two_hop.csv"
    if evaluated_path.exists():
        print(f"Loading existing evaluation from {evaluated_path} ...")
        evaluated = pd.read_csv(evaluated_path, index_col=0)
    else:
        print("Evaluating base model ...")
        raw = pd.read_csv(args.input, index_col=0)
        fn_kwargs = {"model": model, "tokenizer": tokenizer}
        hf = Dataset.from_pandas(raw, preserve_index=False)
        hf = hf.map(get_answers, fn_kwargs={"key": "e2"})
        hf = hf.map(get_answers, fn_kwargs={"key": "e3"})
        hf = hf.map(eval_first_hop,         batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf = hf.map(eval_second_hop,        batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf = hf.map(eval_composition,       batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf = hf.map(eval_entity_shortcut,   batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf = hf.map(eval_relation_shortcut, batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        evaluated = hf.to_pandas().drop(["e2_answers", "e3_answers"], axis=1).reset_index(drop=True)
        evaluated.to_csv(evaluated_path)
        print(f"Saved {evaluated_path}")

    if args.max_examples:
        evaluated = evaluated.head(args.max_examples)

    # ── Create subsets ────────────────────────────────────────────────────────
    subsets = create_subsets(evaluated, data_dir)

    # ── Run analyses per subset ───────────────────────────────────────────────
    for subset_name, subset_df in subsets.items():
        subset_dir = data_dir / f"composition_{subset_name}"

        print(f"\n{'='*60}")
        print(f"Subset: {subset_name} ({len(subset_df)} examples)")
        print(f"{'='*60}")

        subset_df = subset_df.copy()
        subset_df["e2_answers"] = subset_df.apply(lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
        subset_df["e3_answers"] = subset_df.apply(lambda r: get_answers(r, "e3")["e3_answers"], axis=1)
        hf_dataset = Dataset.from_pandas(subset_df, preserve_index=False)

        # ── Method 1: Sublayer projection — attention ─────────────────────────
        out = subset_dir / "sublayer_projection" / "attention_projections.csv"
        if out.exists():
            print(f"\n=== Method 1/4: Sublayer projection (attention) — SKIP (exists) ===")
        else:
            print("\n=== Method 1/4: Sublayer projection (attention) ===")
            sublayer_attn = to_sublayer_df(hf_dataset.map(
                project_attention,
                fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
            ))
            out.parent.mkdir(parents=True, exist_ok=True)
            sublayer_attn.to_csv(out, escapechar="\\")
            print(f"Saved {out}")

        # ── Method 2: Sublayer projection — MLP ───────────────────────────────
        out = subset_dir / "sublayer_projection" / "mlp_projections.csv"
        if out.exists():
            print(f"\n=== Method 2/4: Sublayer projection (MLP) — SKIP (exists) ===")
        else:
            print("\n=== Method 2/4: Sublayer projection (MLP) ===")
            sublayer_mlp = to_sublayer_df(hf_dataset.map(
                project_attention,
                fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
            ))
            out.parent.mkdir(parents=True, exist_ok=True)
            sublayer_mlp.to_csv(out, escapechar="\\")
            print(f"Saved {out}")

        # ── Method 3: Attention knockout ──────────────────────────────────────
        out = subset_dir / "attention_knockout" / "last_e1_attention_knockout_k3_composition.csv"
        if out.exists():
            print(f"\n=== Method 3/4: Attention knockout — SKIP (exists) ===")
        else:
            print("\n=== Method 3/4: Attention knockout ===")
            knockout = hf_dataset.map(
                generate_answer_with_attention_knockout,
                fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                           "knockout_source": "last", "knockout_target": "e1", "k": 3,
                           "prompt": "composition"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
            ).to_pandas()
            out.parent.mkdir(parents=True, exist_ok=True)
            knockout.to_csv(out, escapechar="\\")
            print(f"Saved {out}")

        # ── Method 4: PatchScopes (optional, slow) ────────────────────────────
        if not args.skip_patchscopes:
            dataset_ps = subset_df.copy()
            dataset_ps["target_prompt"] = TARGET_PROMPT
            hf_ps = Dataset.from_pandas(dataset_ps, preserve_index=False)
            for source in ["e1", "last"]:
                out = subset_dir / "entity_description" / f"{source}_entity_description_s0.csv"
                if out.exists():
                    print(f"\n=== Method 4/4: PatchScopes source={source} — SKIP (exists) ===")
                    continue
                print(f"\n=== Method 4/4: PatchScopes source={source} ===")
                patch = hf_ps.map(
                    generate_entity_description_with_patching,
                    fn_kwargs={"model": model, "tokenizer": tokenizer,
                               "source": source, "do_sample": False},
                    batched=True, batch_size=args.batch_size,
                    remove_columns=hf_ps.column_names,
                ).to_pandas()
                out.parent.mkdir(parents=True, exist_ok=True)
                patch.to_csv(out, escapechar="\\")
                print(f"Saved {out}")

    print(f"\nDone. Run: python analysis/analyze_tables.py --subsets "
          f"correct:{data_dir}/composition_correct "
          f"incorrect:{data_dir}/composition_incorrect")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",    default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--input",    default="datasets/two_hop.csv",
                        help="Raw (unevaluated) two_hop.csv dataset")
    parser.add_argument("--data-dir", default="datasets/meta-llama/Meta-Llama-3-8B",
                        help="Output directory; evaluated two_hop.csv and subsets saved here")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-patchscopes", action="store_true",
                        help="Skip patchscopes (method 4) to save time")
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
