# %%
"""
Apply IFMET to all incorrect examples, then run the same sublayer projection,
attention knockout, and patchscopes analyses as run_edited_analysis.py.

IFMET uses MEMIT in two phases:
  Phase 1 (shallow layers): edit individual facts  e1→e2 and e2→e3
  Phase 2 (deep layers):   edit the indirect composition question (source_prompt→e3)

This mirrors edit_ifmet() in edit_utils.py but applies all edits in one batch
(copy=False, return_orig_weights=False) so that weights persist for analysis.

Usage:
    python analysis/run_ifmet_analysis.py \\
        [--model  meta-llama/Meta-Llama-3-8B] \\
        [--hparams-shallow EasyEdit/hparams/MEMIT/llama3-8b-shallow.yaml] \\
        [--hparams-deep    EasyEdit/hparams/MEMIT/llama3-8b-deep.yaml] \\
        [--input  datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/two_hop.csv] \\
        [--output datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_ifmet] \\
        [--max-examples N] [--skip-patchscopes] [--device 0]
"""

import os
import sys

os.environ["USE_TF"] = "0"
sys.path.insert(0, ".")
sys.path.insert(0, "analysis")

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

from EasyEdit.easyeditor import MEMITHyperParams
from EasyEdit.easyeditor.util import nethook
from EasyEdit.easyeditor.util.alg_dict import ALG_DICT

from evaluate_dataset import (
    eval_composition, eval_first_hop, eval_second_hop,
    eval_entity_shortcut, eval_relation_shortcut,
)
from generate_entity_description import generate_entity_description_with_patching
from knockout_attention import generate_answer_with_attention_knockout
from project_sublayer import project_attention
from utils import get_answers, load_model, load_tokenizer

# Reuse subset splitting and save helpers from run_edited_analysis
from run_alphaedit_analysis import split_subsets, to_sublayer_df, save_by_subset


# %% IFMET edit helpers

def build_standard_requests(dataset: pd.DataFrame) -> list:
    """Phase 1: single-hop fact edits (same as AlphaEdit)."""
    requests = []
    for _, row in dataset.iterrows():
        requests.append({"prompt": row["r1_template"], "subject": row["e1_label"],
                         "target_new": row["e2_label"]})
        requests.append({"prompt": row["r2_template"], "subject": row["e2_label"],
                         "target_new": row["e3_label"]})
    return requests


def build_indirect_requests(dataset: pd.DataFrame) -> list:
    """Phase 2: indirect composition edit (source_prompt → e3_label).
    Subject = full prompt, matching edit_ifmet()'s convention."""
    requests = []
    for _, row in dataset.iterrows():
        prompt = row["source_prompt"]
        requests.append({"prompt": prompt, "subject": prompt,
                         "target_new": row["e3_label"]})
    return requests


def save_checkpoint(model, hparams_s, hparams_d, ckpt_path: Path):
    """Save edited weights from both phases."""
    weight_names = set()
    for hparams in [hparams_s, hparams_d]:
        for layer in hparams.layers:
            weight_names.add(f"{hparams.rewrite_module_tmp.format(layer)}.weight")
    weights = {k: nethook.get_parameter(model, k).detach().cpu() for k in weight_names}
    torch.save(weights, ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


def load_checkpoint(model, ckpt_path: Path, device: str):
    """Restore edited weights from checkpoint."""
    weights = torch.load(ckpt_path, map_location=device)
    with torch.no_grad():
        for k, v in weights.items():
            nethook.get_parameter(model, k)[...] = v.to(device)
    print(f"Loaded edited weights from {ckpt_path}")


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

    # ── IFMET hparams ─────────────────────────────────────────────────────────
    hparams_s  = MEMITHyperParams.from_hparams(args.hparams_shallow)
    hparams_d  = MEMITHyperParams.from_hparams(args.hparams_deep)
    apply_algo = ALG_DICT[hparams_s.alg_name]   # MEMIT

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = pd.read_csv(args.input, index_col=0)
    if args.max_examples:
        dataset = dataset.head(args.max_examples)
    dataset["e2_answers"] = dataset.apply(lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
    dataset["e3_answers"] = dataset.apply(lambda r: get_answers(r, "e3")["e3_answers"], axis=1)

    # ── Apply IFMET (or load checkpoint) ──────────────────────────────────────
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device)
    else:
        standard_requests = build_standard_requests(dataset)
        indirect_requests = build_indirect_requests(dataset)

        print(f"Phase 1: MEMIT shallow — {len(standard_requests)} fact edits ...")
        hparams_s.batch_size = len(standard_requests)
        apply_algo(model, tokenizer, standard_requests, hparams_s,
                   copy=False, return_orig_weights=False)

        print(f"Phase 2: MEMIT deep — {len(indirect_requests)} indirect edits ...")
        hparams_d.batch_size = len(indirect_requests)
        apply_algo(model, tokenizer, indirect_requests, hparams_d,
                   copy=False, return_orig_weights=False)

        save_checkpoint(model, hparams_s, hparams_d, ckpt_path)

    # ── Evaluate edited model (if not already done) ───────────────────────────
    evaluated_path = out_dir / "evaluated_two_hop.csv"
    if not evaluated_path.exists():
        print("Running evaluation on edited model ...")
        fn_kwargs = {"model": model, "tokenizer": tokenizer}
        hf_eval = Dataset.from_pandas(dataset, preserve_index=False)
        hf_eval = hf_eval.map(eval_first_hop,         batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_eval = hf_eval.map(eval_second_hop,        batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_eval = hf_eval.map(eval_composition,       batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_eval = hf_eval.map(eval_entity_shortcut,   batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_eval = hf_eval.map(eval_relation_shortcut,  batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        evaluated = hf_eval.to_pandas().drop(["e2_answers", "e3_answers"], axis=1).reset_index(drop=True)
        evaluated.to_csv(evaluated_path)
        print(f"Saved {evaluated_path}")
    else:
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

    # Analyses only for now_correct and still_incorrect (inconsistent excluded)
    analysis_subsets = {k: v for k, v in subsets.items() if k != "still_inconsistent"}

    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    def outputs_exist(subdir, filename):
        return all((out_dir / name / subdir / filename).exists() for name in analysis_subsets)

    # ── Method 1: Sublayer projection — attention ─────────────────────────────
    if outputs_exist("sublayer_projection", "attention_projections.csv"):
        print("\n=== Method 1/4: Sublayer projection (attention) — SKIP (exists) ===")
    else:
        print("\n=== Method 1/4: Sublayer projection (attention) ===")
        sublayer_attn = to_sublayer_df(hf_dataset.map(
            project_attention,
            fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
            batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
        ))
        save_by_subset(sublayer_attn, analysis_subsets, out_dir,
                       "sublayer_projection", "attention_projections.csv", ["id", "layer"])

    # ── Method 2: Sublayer projection — MLP ───────────────────────────────────
    if outputs_exist("sublayer_projection", "mlp_projections.csv"):
        print("\n=== Method 2/4: Sublayer projection (MLP) — SKIP (exists) ===")
    else:
        print("\n=== Method 2/4: Sublayer projection (MLP) ===")
        sublayer_mlp = to_sublayer_df(hf_dataset.map(
            project_attention,
            fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
            batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
        ))
        save_by_subset(sublayer_mlp, analysis_subsets, out_dir,
                       "sublayer_projection", "mlp_projections.csv", ["id", "layer"])

    # ── Method 3: Attention knockout ──────────────────────────────────────────
    if outputs_exist("attention_knockout", "last_e1_attention_knockout_k3_composition.csv"):
        print("\n=== Method 3/4: Attention knockout — SKIP (exists) ===")
    else:
        print("\n=== Method 3/4: Attention knockout ===")
        knockout = hf_dataset.map(
            generate_answer_with_attention_knockout,
            fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                       "knockout_source": "last", "knockout_target": "e1", "k": 3,
                       "prompt": "composition"},
            batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
        ).to_pandas()
        save_by_subset(knockout, analysis_subsets, out_dir,
                       "attention_knockout", "last_e1_attention_knockout_k3_composition.csv",
                       ["id", "layer"])

    # ── Method 4: PatchScopes (optional, slow) ────────────────────────────────
    if not args.skip_patchscopes:
        target_prompt = (
            "Syria: Syria is a country in the Middle East, "
            "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, "
            "Samsung: Samsung is a South Korean multinational corporation, "
            "x"
        )
        dataset_ps = dataset.copy()
        dataset_ps["target_prompt"] = target_prompt
        hf_ps = Dataset.from_pandas(dataset_ps, preserve_index=False)
        for source in ["e1", "last"]:
            if outputs_exist("entity_description", f"{source}_entity_description_s0.csv"):
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
            save_by_subset(patch, analysis_subsets, out_dir,
                           "entity_description", f"{source}_entity_description_s0.csv",
                           ["id", "source_layer", "target_layer"])

    print(f"\nDone. Run: python analysis/analyze_tables.py --subsets pre-edit:<base_dir> now_correct:{out_dir}/now_correct ...")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",   default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--hparams-shallow",
                        default="EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml",
                        help="MEMIT hparams for phase 1 (standard fact edits, shallow layers)")
    parser.add_argument("--hparams-deep",
                        default="EasyEdit/hparams/IFMET/llama3-8b-deep.yaml",
                        help="MEMIT hparams for phase 2 (indirect composition edits, deep layers)")
    parser.add_argument("--input",
                        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/two_hop.csv")
    parser.add_argument("--output",
                        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_ifmet")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-patchscopes", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    main(parser.parse_args())
