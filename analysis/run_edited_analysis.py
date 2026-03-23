"""
Apply AlphaEdit to all incorrect examples in one batch, then run sublayer
projection, attention knockout, and patchscopes analyses on the fully-edited
model.

All facts (2 per example: e1→e2 and e2→e3) are passed to AlphaEdit in a
single call, resulting in one matrix solve instead of N. Edited weights are
saved to a checkpoint so re-runs skip the editing phase entirely.

Output CSVs are written to <output_dir>/ in the same format as the existing
composition_incorrect/ directory, so analyze_subset.py can be run afterward
to assemble layers.csv for comparison with the baseline.

Note: AlphaEdit computes a null-space projection matrix P and Wikipedia
covariance stats on first run if not already cached (see P_loc in the hparams
YAML). This is slow (~minutes) but only happens once.

Usage:
    python analysis/run_edited_analysis.py \\
        [--model  meta-llama/Meta-Llama-3-8B] \\
        [--hparams EasyEdit/hparams/AlphaEdit/llama3-8b.yaml] \\
        [--input  datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/two_hop.csv] \\
        [--output datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited] \\
        [--max-examples N] [--skip-patchscopes] [--device 0]

After this script completes:
    python analysis/analyze_subset.py <output_dir>
    python analysis/analyze_mechanism.py --incorrect <output_dir>/layers.csv
"""

import sys
sys.path.insert(0, "analysis")

import argparse
import shutil
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


def build_requests(dataset: pd.DataFrame) -> list:
    """Build one AlphaEdit request per hop for every example."""
    requests = []
    for _, row in dataset.iterrows():
        requests.append({"prompt": row["r1_template"], "subject": row["e1_label"], "target_new": row["e2_label"]})
        requests.append({"prompt": row["r2_template"], "subject": row["e2_label"], "target_new": row["e3_label"]})
    return requests


def save_checkpoint(model, hparams, ckpt_path: Path):
    """Save edited weights to disk so editing can be skipped on rerun."""
    weight_names = [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]
    weights = {k: nethook.get_parameter(model, k).detach().cpu() for k in weight_names}
    torch.save(weights, ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


def load_checkpoint(model, ckpt_path: Path, device: str):
    """Restore edited weights from checkpoint, skipping the editing phase."""
    weights = torch.load(ckpt_path, map_location=device)
    with torch.no_grad():
        for k, v in weights.items():
            nethook.get_parameter(model, k)[...] = v.to(device)
    print(f"Loaded edited weights from {ckpt_path}")


def main(args):
    out_dir  = Path(args.output)
    ckpt_path = out_dir / "editing_checkpoint.pt"
    device    = f"cuda:{args.device}"

    # ── Model & tokenizer ─────────────────────────────────────────────────────
    print(f"Loading model {args.model} ...")
    model = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)

    # ── AlphaEdit ─────────────────────────────────────────────────────────────
    hparams = AlphaEditHyperParams.from_hparams(args.hparams)
    apply_algo = ALG_DICT["AlphaEdit"]

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = pd.read_csv(args.input, index_col=0)
    if args.max_examples:
        dataset = dataset.head(args.max_examples)

    # Recompute answers as proper Python lists (CSV may store numpy-style strings)
    dataset["e2_answers"] = dataset.apply(
        lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
    dataset["e3_answers"] = dataset.apply(
        lambda r: get_answers(r, "e3")["e3_answers"], axis=1)

    # ── Phase 1: apply all edits in one batch ────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device)
    else:
        requests = build_requests(dataset)
        print(f"Applying AlphaEdit ({len(requests)} facts across {len(dataset)} examples)...")
        apply_algo(model, tokenizer, requests, hparams, copy=False, return_orig_weights=False)
        save_checkpoint(model, hparams, ckpt_path)

    # ── Phase 2: run analyses on the fully-edited model ───────────────────────
    print("Running analyses on edited model...")
    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    sublayer_attn = hf_dataset.map(
        project_attention,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
        batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
    ).to_pandas()

    sublayer_mlp = hf_dataset.map(
        project_attention,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
        batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
    ).to_pandas()

    knockout = hf_dataset.map(
        generate_answer_with_attention_knockout,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                   "knockout_source": "last", "knockout_target": "e1", "k": 3,
                   "prompt": "composition"},
        batched=True, batch_size=args.batch_size, remove_columns=hf_dataset.column_names,
    ).to_pandas()

    if not args.skip_patchscopes:
        dataset_e1   = dataset.copy()
        dataset_last = dataset.copy()
        dataset_e1["target_prompt"]   = create_target_prompt(dataset, "e1",   "original")
        dataset_last["target_prompt"] = create_target_prompt(dataset, "last", "original")

        hf_dataset_e1   = Dataset.from_pandas(dataset_e1,   preserve_index=False)
        hf_dataset_last = Dataset.from_pandas(dataset_last, preserve_index=False)

        patch_e1 = hf_dataset_e1.map(
            activation_patching,
            fn_kwargs={"model": model, "tokenizer": tokenizer,
                       "source": "e1", "target": "original", "default_decoding": False},
            batched=True, batch_size=args.batch_size,
            remove_columns=hf_dataset_e1.column_names,
        ).to_pandas()

        patch_last = hf_dataset_last.map(
            activation_patching,
            fn_kwargs={"model": model, "tokenizer": tokenizer,
                       "source": "last", "target": "original", "default_decoding": False},
            batched=True, batch_size=args.batch_size,
            remove_columns=hf_dataset_last.column_names,
        ).to_pandas()

    # ── Save results ──────────────────────────────────────────────────────────
    def save_layer_csv(df, subdir, filename):
        path = out_dir / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["id", "layer"]).reset_index(drop=True).to_csv(path, escapechar="\\")
        print(f"Saved {path}")

    def save_patching_csv(df, filename):
        path = out_dir / "activation_patching" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["id", "source_layer", "target_layer"]).reset_index(drop=True).to_csv(path, escapechar="\\")
        print(f"Saved {path}")

    save_layer_csv(sublayer_attn, "sublayer_projection", "attention_projections.csv")
    save_layer_csv(sublayer_mlp,  "sublayer_projection", "mlp_projections.csv")
    save_layer_csv(knockout,      "attention_knockout",  "last_e1_attention_knockout_k3_composition.csv")

    if not args.skip_patchscopes:
        save_patching_csv(patch_e1,   "e1_original_activation_patching.csv")
        save_patching_csv(patch_last, "last_original_activation_patching.csv")

    # Copy dataset CSV so analyze_subset.py can find it
    shutil.copy(args.input, out_dir / "two_hop.csv")

    print(f"\nDone. Next steps:")
    print(f"  python analysis/analyze_subset.py {out_dir}")
    print(f"  python analysis/analyze_mechanism.py --incorrect {out_dir}/layers.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",   default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--hparams", default="EasyEdit/hparams/AlphaEdit/llama3-8b.yaml")
    parser.add_argument("--input",
                        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect/two_hop.csv")
    parser.add_argument("--output",
                        default="datasets/meta-llama/Meta-Llama-3-8B/composition_incorrect_edited")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit to first N examples (useful for testing)")
    parser.add_argument("--skip-patchscopes", action="store_true",
                        help="Skip patchscopes to save time")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    main(parser.parse_args())
