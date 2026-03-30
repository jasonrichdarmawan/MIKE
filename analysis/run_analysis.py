# %%
"""
Run the full analysis pipeline (pre-edit, AlphaEdit, IFMET) for one model.

Loads the model fresh for each method (edits are applied in-place so weights
must be restored between runs).

Usage:
    python analysis/run_analysis.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --methods preedit alphaedit ifmet \\
        [--input      datasets/two_hop.csv] \\
        [--data-dir   datasets/meta-llama/Meta-Llama-3-8B] \\
        [--hparams-alphaedit  EasyEdit/hparams/AlphaEdit/llama3-8b.yaml] \\
        [--hparams-ifmet-shallow EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml] \\
        [--hparams-ifmet-deep   EasyEdit/hparams/IFMET/llama3-8b-deep.yaml] \\
        [--skip-patchscopes] [--max-examples N] [--batch-size 128] \\
        [--eap-top-k 5] [--eap-batch-size 16] [--eap-samples N] [--device 0]
"""

# %%

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
from tqdm import tqdm

from evaluate_dataset import (
    eval_composition, eval_first_hop, eval_second_hop,
    eval_entity_shortcut, eval_relation_shortcut,
)
from generate_entity_description import generate_entity_description_with_patching
from knockout_attention import generate_answer_with_attention_knockout
from project_sublayer import project_attention
from utils import get_answers, load_model, load_tokenizer
from EasyEdit.easyeditor.util import nethook


# %%

TARGET_PROMPT = (
    "Syria: Syria is a country in the Middle East, "
    "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, "
    "Samsung: Samsung is a South Korean multinational corporation, "
    "x"
)


# %% Pre-edit helpers

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

    correct_second_hop = set(
        (r["e2_label"], r["r2_type"], r["e3_label"])
        for r in tqdm(df_correct.to_dict("records"), desc="Building correct second-hop set")
    )
    df_incorrect = df_incorrect.copy()
    df_incorrect["is_correct_second_hop"] = df_incorrect.apply(
        lambda r: (r["e2_label"], r["r2_type"], r["e3_label"]) in correct_second_hop, axis=1
    )
    df_inconsistent  = df_incorrect[df_incorrect["is_correct_second_hop"]]
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


# %% Post-edit shared helpers

def filter_multi_target(df: pd.DataFrame, log_path: Path = None) -> pd.DataFrame:
    """Remove rows where any (entity, relation) pair has more than one valid target.

    Such ambiguous pairs cause false positives in still_incorrect / still_inconsistent /
    broken_hops: the model may produce a different but equally valid answer that we
    would wrongly label as incorrect.

    If log_path is given, saves removed rows to that CSV for inspection.
    """
    t1 = df[["e1", "r1", "e2"]].rename(columns={"e1": "entity", "r1": "relation", "e2": "target"})
    t2 = df[["e2", "r2", "e3"]].rename(columns={"e2": "entity", "r2": "relation", "e3": "target"})
    target_counts = pd.concat([t1, t2]).groupby(["entity", "relation"])["target"].nunique()
    valid_pairs = target_counts[target_counts == 1].index
    idx1 = pd.MultiIndex.from_arrays([df["e1"], df["r1"]])
    idx2 = pd.MultiIndex.from_arrays([df["e2"], df["r2"]])
    mask = idx1.isin(valid_pairs) & idx2.isin(valid_pairs)
    filtered = df[mask]
    removed_df = df[~mask]
    if len(removed_df):
        print(f"  filter_multi_target: removed {len(removed_df)} rows ({len(filtered)} remaining)")
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            removed_df.to_csv(log_path)
            print(f"  Removed rows saved to {log_path}")
    return filtered

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


def split_subsets(evaluated: pd.DataFrame) -> dict:
    """
    Split post-edit evaluated dataset into the three outcome subsets.

    still_inconsistent: composition still fails but the (e2, r2, e3) second
    hop appears among now_correct examples — meaning the bridge entity is
    decodable in some contexts, pointing to a composition failure.
    """
    passes_hops   = evaluated["first_hop_correct"] & evaluated["second_hop_correct"]
    broken_hops   = evaluated[~passes_hops]
    now_correct   = evaluated[evaluated["composition_correct"] & passes_hops]
    still_failed  = evaluated[~evaluated["composition_correct"] & passes_hops]

    correct_second_hops = set(
        zip(now_correct["e2_label"], now_correct["r2_type"], now_correct["e3_label"])
    )
    still_inconsistent = still_failed[still_failed.apply(
        lambda r: (r["e2_label"], r["r2_type"], r["e3_label"]) in correct_second_hops, axis=1
    )]
    still_incorrect = still_failed[~still_failed.index.isin(still_inconsistent.index)]

    print(f"  now_correct:         {len(now_correct)}")
    print(f"  still_incorrect:     {len(still_incorrect)}")
    print(f"  still_inconsistent:  {len(still_inconsistent)}")
    print(f"  broken_hops:         {len(broken_hops)}")
    return {
        "now_correct":        now_correct,
        "still_incorrect":    still_incorrect,
        "still_inconsistent": still_inconsistent,
        "broken_hops":        broken_hops,
    }


# %% Per-method runners

def run_preedit(args):
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    model     = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"

    # ── Evaluate base model ───────────────────────────────────────────────────
    evaluated_path = data_dir / "two_hop.csv"
    if not args.force and evaluated_path.exists():
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

    subsets = create_subsets(evaluated, data_dir)

    ANALYSE_SUBSETS = {"correct", "incorrect"}
    for subset_name, subset_df in subsets.items():
        if subset_name not in ANALYSE_SUBSETS:
            print(f"\nSkipping analyses for subset '{subset_name}'")
            continue
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
        if not args.force and out.exists():
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
        if not args.force and out.exists():
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
        if not args.force and out.exists():
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
                if not args.force and out.exists():
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


def run_alphaedit(args):
    data_dir      = Path(args.data_dir)
    incorrect_dir = data_dir / "composition_incorrect"
    out_dir       = data_dir / "composition_incorrect_alphaedit"
    device        = f"cuda:{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    model     = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)

    from EasyEdit.easyeditor import AlphaEditHyperParams
    from EasyEdit.easyeditor.util import nethook
    from EasyEdit.easyeditor.util.alg_dict import ALG_DICT

    hparams    = AlphaEditHyperParams.from_hparams(args.hparams_alphaedit)
    apply_algo = ALG_DICT["AlphaEdit"]

    dataset = pd.read_csv(str(incorrect_dir / "two_hop.csv"), index_col=0)
    dataset = filter_multi_target(dataset, log_path=incorrect_dir / "filtered_multi_target.csv")
    if args.max_examples:
        dataset = dataset.head(args.max_examples)
    dataset["e2_answers"] = dataset.apply(lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
    dataset["e3_answers"] = dataset.apply(lambda r: get_answers(r, "e3")["e3_answers"], axis=1)

    weight_names = [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]

    orig_weights = {k: nethook.get_parameter(model, k).detach().clone().cpu()
                    for k in weight_names}

    n_batches     = (len(dataset) + args.batch_size - 1) // args.batch_size
    all_evaluated     = []
    all_sublayer_attn = []
    all_sublayer_mlp  = []
    all_knockout      = []
    all_patch         = {source: [] for source in ["e1", "last"]}

    for b, start_idx in enumerate(range(0, len(dataset), args.batch_size)):
        batch_df = dataset.iloc[start_idx:start_idx + args.batch_size].copy()
        print(f"\n--- AlphaEdit Batch {b+1}/{n_batches} ({len(batch_df)} examples) ---")

        # Edit — one apply_algo call for the whole batch so the null-space
        # projection is estimated over all batch facts at once.
        batch_requests = []
        for _, row in batch_df.iterrows():
            batch_requests.append({"prompt": row["r1_template"], "subject": row["e1_label"], "target_new": row["e2_label"]})
            batch_requests.append({"prompt": row["r2_template"], "subject": row["e2_label"], "target_new": row["e3_label"]})
        hparams.batch_size = len(batch_requests)
        apply_algo(model, tokenizer, batch_requests, hparams, copy=False, return_orig_weights=False)
        print(f"  Batch edited ({len(batch_requests)} requests).")

        # Evaluate
        print(f"  [Phase: Eval] Batch {b+1}/{n_batches}")
        fn_kwargs = {"model": model, "tokenizer": tokenizer}
        hf_b = Dataset.from_pandas(batch_df, preserve_index=False)
        hf_b = hf_b.map(eval_first_hop,         batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_second_hop,        batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_composition,       batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_entity_shortcut,   batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_relation_shortcut, batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        all_evaluated.append(hf_b.to_pandas().drop(["e2_answers", "e3_answers"], axis=1).reset_index(drop=True))

        # Analyze
        if not args.skip_analysis:
            hf_analysis = Dataset.from_pandas(batch_df, preserve_index=False)
            print(f"  [Phase: Sublayer-Attn] Batch {b+1}/{n_batches}")
            all_sublayer_attn.append(to_sublayer_df(hf_analysis.map(
                project_attention, fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names)))
            print(f"  [Phase: Sublayer-MLP] Batch {b+1}/{n_batches}")
            all_sublayer_mlp.append(to_sublayer_df(hf_analysis.map(
                project_attention, fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names)))
            print(f"  [Phase: Knockout] Batch {b+1}/{n_batches}")
            all_knockout.append(hf_analysis.map(
                generate_answer_with_attention_knockout,
                fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                           "knockout_source": "last", "knockout_target": "e1", "k": 3, "prompt": "composition"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names).to_pandas())
            if not args.skip_patchscopes:
                print(f"  [Phase: PatchScopes] Batch {b+1}/{n_batches}")
                batch_ps = batch_df.copy()
                batch_ps["target_prompt"] = TARGET_PROMPT
                hf_ps = Dataset.from_pandas(batch_ps, preserve_index=False)
                for source in ["e1", "last"]:
                    all_patch[source].append(hf_ps.map(
                        generate_entity_description_with_patching,
                        fn_kwargs={"model": model, "tokenizer": tokenizer, "source": source, "do_sample": False},
                        batched=True, batch_size=args.batch_size, remove_columns=hf_ps.column_names).to_pandas())

        # Restore
        print(f"  [Phase: Restore] Batch {b+1}/{n_batches}")
        with torch.no_grad():
            for k, v in orig_weights.items():
                nethook.get_parameter(model, k)[...] = v.to(device)
        torch.cuda.empty_cache()
        print(f"  Weights restored to original.")

    evaluated_path = out_dir / "evaluated_two_hop.csv"
    evaluated = pd.concat(all_evaluated).reset_index(drop=True)
    evaluated.to_csv(evaluated_path)
    print(f"Saved {evaluated_path}")

    subsets = split_subsets(evaluated)
    for name, df in subsets.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / name / "two_hop.csv")
        print(f"Saved {out_dir / name / 'two_hop.csv'}")

    if not args.skip_analysis:
        analysis_subsets = subsets
        save_by_subset(pd.concat(all_sublayer_attn), analysis_subsets, out_dir,
                       "sublayer_projection", "attention_projections.csv", ["id", "layer"])
        save_by_subset(pd.concat(all_sublayer_mlp), analysis_subsets, out_dir,
                       "sublayer_projection", "mlp_projections.csv", ["id", "layer"])
        save_by_subset(pd.concat(all_knockout), analysis_subsets, out_dir,
                       "attention_knockout", "last_e1_attention_knockout_k3_composition.csv", ["id", "layer"])
        if not args.skip_patchscopes:
            for source in ["e1", "last"]:
                if all_patch[source]:
                    save_by_subset(pd.concat(all_patch[source]), analysis_subsets, out_dir,
                                   "entity_description", f"{source}_entity_description_s0.csv",
                                   ["id", "source_layer", "target_layer"])

    print(f"\nDone. Run: python analysis/analyze_tables.py --subsets "
          f"pre-edit:{data_dir}/composition_incorrect "
          f"now_correct:{out_dir}/now_correct "
          f"still_incorrect:{out_dir}/still_incorrect")


def run_eap_alphaedit(args):
    """AlphaEdit with layers chosen by EAP attribution on the incorrect dataset."""
    data_dir      = Path(args.data_dir)
    incorrect_dir = data_dir / "composition_incorrect"
    out_dir       = data_dir / "composition_incorrect_eap_alphaedit"
    device        = f"cuda:{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    model     = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)

    from EasyEdit.easyeditor import AlphaEditHyperParams
    from EasyEdit.easyeditor.util import nethook
    from EasyEdit.easyeditor.util.alg_dict import ALG_DICT
    from transformers_eap import EAPGraph
    from transformers_eap.eap_graph import get_nll_metric

    hparams    = AlphaEditHyperParams.from_hparams(args.hparams_alphaedit)
    apply_algo = ALG_DICT["AlphaEdit"]

    dataset = pd.read_csv(str(incorrect_dir / "two_hop.csv"), index_col=0)
    dataset = filter_multi_target(dataset, log_path=incorrect_dir / "filtered_multi_target.csv")
    if args.max_examples:
        dataset = dataset.head(args.max_examples)
    dataset["e2_answers"] = dataset.apply(lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
    dataset["e3_answers"] = dataset.apply(lambda r: get_answers(r, "e3")["e3_answers"], axis=1)

    # ── EAP: identify top-k MLP layers ──────────────────────────────────────
    print(f"\n[EAP] Identifying top-{args.eap_top_k} MLP layers on r1+r2 prompts ...")
    eap_data = dataset.head(args.eap_samples) if args.eap_samples else dataset
    n_layers = model.config.num_hidden_layers
    accumulated = torch.zeros(n_layers, device=device)

    eap = EAPGraph(model)
    for start in range(0, len(eap_data), args.eap_batch_size):
        batch = eap_data.iloc[start:start + args.eap_batch_size]

        # Build r1 and r2 prompts together
        prompts, ans_labels = [], []
        for _, row in batch.iterrows():
            prompts.append(row["r1_template"].replace("{}", row["e1_label"]))
            ans_labels.append(row["e2_label"])
            prompts.append(row["r2_template"].replace("{}", row["e2_label"]))
            ans_labels.append(row["e3_label"])

        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        ans_ids = torch.tensor([
            tokenizer(lbl, add_special_tokens=False)["input_ids"][0]
            for lbl in ans_labels
        ], device=device)

        with torch.enable_grad():
            scores = eap.attribute(
                enc["input_ids"], enc["attention_mask"],
                metric_fn=get_nll_metric(ans_ids),
                return_per_token_scores=False,
                return_per_head_attribution=False,
            )
        eap.reset_hooks()

        for l in range(n_layers):
            key = f"blocks.{l}.hook_mlp_in"
            if key in scores and isinstance(scores[key], torch.Tensor):
                accumulated[l] += scores[key].abs().sum().detach()

    top_layers = accumulated.topk(args.eap_top_k).indices.tolist()
    top_layers.sort()
    print(f"[EAP] Top-{args.eap_top_k} layers: {top_layers}")
    hparams.layers = top_layers
    torch.cuda.empty_cache()
    # ────────────────────────────────────────────────────────────────────────

    weight_names = [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]
    orig_weights = {k: nethook.get_parameter(model, k).detach().clone().cpu()
                    for k in weight_names}

    n_batches         = (len(dataset) + args.batch_size - 1) // args.batch_size
    all_evaluated     = []
    all_sublayer_attn = []
    all_sublayer_mlp  = []
    all_knockout      = []
    all_patch         = {source: [] for source in ["e1", "last"]}

    for b, start_idx in enumerate(range(0, len(dataset), args.batch_size)):
        batch_df = dataset.iloc[start_idx:start_idx + args.batch_size].copy()
        print(f"\n--- EAP-AlphaEdit Batch {b+1}/{n_batches} ({len(batch_df)} examples) ---")

        batch_requests = []
        for _, row in batch_df.iterrows():
            batch_requests.append({"prompt": row["r1_template"], "subject": row["e1_label"], "target_new": row["e2_label"]})
            batch_requests.append({"prompt": row["r2_template"], "subject": row["e2_label"], "target_new": row["e3_label"]})
        hparams.batch_size = len(batch_requests)
        apply_algo(model, tokenizer, batch_requests, hparams, copy=False, return_orig_weights=False)
        print(f"  Batch edited ({len(batch_requests)} requests, layers={top_layers}).")

        # Evaluate
        print(f"  [Phase: Eval] Batch {b+1}/{n_batches}")
        fn_kwargs = {"model": model, "tokenizer": tokenizer}
        hf_b = Dataset.from_pandas(batch_df, preserve_index=False)
        hf_b = hf_b.map(eval_first_hop,         batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_second_hop,        batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_composition,       batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_entity_shortcut,   batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_relation_shortcut, batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        all_evaluated.append(hf_b.to_pandas().drop(["e2_answers", "e3_answers"], axis=1).reset_index(drop=True))

        # Analyze
        if not args.skip_analysis:
            hf_analysis = Dataset.from_pandas(batch_df, preserve_index=False)
            print(f"  [Phase: Sublayer-Attn] Batch {b+1}/{n_batches}")
            all_sublayer_attn.append(to_sublayer_df(hf_analysis.map(
                project_attention, fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names)))
            print(f"  [Phase: Sublayer-MLP] Batch {b+1}/{n_batches}")
            all_sublayer_mlp.append(to_sublayer_df(hf_analysis.map(
                project_attention, fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names)))
            print(f"  [Phase: Knockout] Batch {b+1}/{n_batches}")
            all_knockout.append(hf_analysis.map(
                generate_answer_with_attention_knockout,
                fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                           "knockout_source": "last", "knockout_target": "e1", "k": 3, "prompt": "composition"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names).to_pandas())
            if not args.skip_patchscopes:
                print(f"  [Phase: PatchScopes] Batch {b+1}/{n_batches}")
                batch_ps = batch_df.copy()
                batch_ps["target_prompt"] = TARGET_PROMPT
                hf_ps = Dataset.from_pandas(batch_ps, preserve_index=False)
                for source in ["e1", "last"]:
                    all_patch[source].append(hf_ps.map(
                        generate_entity_description_with_patching,
                        fn_kwargs={"model": model, "tokenizer": tokenizer, "source": source, "do_sample": False},
                        batched=True, batch_size=args.batch_size, remove_columns=hf_ps.column_names).to_pandas())

        # Restore
        print(f"  [Phase: Restore] Batch {b+1}/{n_batches}")
        with torch.no_grad():
            for k, v in orig_weights.items():
                nethook.get_parameter(model, k)[...] = v.to(device)
        torch.cuda.empty_cache()
        print(f"  Weights restored to original.")

    evaluated_path = out_dir / "evaluated_two_hop.csv"
    evaluated = pd.concat(all_evaluated).reset_index(drop=True)
    evaluated.to_csv(evaluated_path)
    print(f"Saved {evaluated_path}")

    subsets = split_subsets(evaluated)
    for name, df in subsets.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / name / "two_hop.csv")
        print(f"Saved {out_dir / name / 'two_hop.csv'}")

    if not args.skip_analysis:
        analysis_subsets = subsets
        save_by_subset(pd.concat(all_sublayer_attn), analysis_subsets, out_dir,
                       "sublayer_projection", "attention_projections.csv", ["id", "layer"])
        save_by_subset(pd.concat(all_sublayer_mlp), analysis_subsets, out_dir,
                       "sublayer_projection", "mlp_projections.csv", ["id", "layer"])
        save_by_subset(pd.concat(all_knockout), analysis_subsets, out_dir,
                       "attention_knockout", "last_e1_attention_knockout_k3_composition.csv", ["id", "layer"])
        if not args.skip_patchscopes:
            for source in ["e1", "last"]:
                if all_patch[source]:
                    save_by_subset(pd.concat(all_patch[source]), analysis_subsets, out_dir,
                                   "entity_description", f"{source}_entity_description_s0.csv",
                                   ["id", "source_layer", "target_layer"])

    print(f"\nDone. Run: python analysis/analyze_tables.py --subsets "
          f"pre-edit:{data_dir}/composition_incorrect "
          f"now_correct:{out_dir}/now_correct "
          f"still_incorrect:{out_dir}/still_incorrect")


def run_ifmet(args):
    data_dir      = Path(args.data_dir)
    incorrect_dir = data_dir / "composition_incorrect"
    out_dir       = data_dir / "composition_incorrect_ifmet"
    device        = f"cuda:{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    model     = load_model(args.model)
    model.eval()
    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"

    from EasyEdit.easyeditor import MEMITHyperParams
    from EasyEdit.easyeditor.util import nethook
    from EasyEdit.easyeditor.util.alg_dict import ALG_DICT

    hparams_s  = MEMITHyperParams.from_hparams(args.hparams_ifmet_shallow)
    hparams_d  = MEMITHyperParams.from_hparams(args.hparams_ifmet_deep)
    apply_algo = ALG_DICT[hparams_s.alg_name]  # MEMIT

    dataset = pd.read_csv(str(incorrect_dir / "two_hop.csv"), index_col=0)
    dataset = filter_multi_target(dataset, log_path=incorrect_dir / "filtered_multi_target.csv")
    if args.max_examples:
        dataset = dataset.head(args.max_examples)
    dataset["e2_answers"] = dataset.apply(lambda r: get_answers(r, "e2")["e2_answers"], axis=1)
    dataset["e3_answers"] = dataset.apply(lambda r: get_answers(r, "e3")["e3_answers"], axis=1)

    weight_names = set()
    for hp in [hparams_s, hparams_d]:
        for layer in hp.layers:
            weight_names.add(f"{hp.rewrite_module_tmp.format(layer)}.weight")

    orig_weights = {k: nethook.get_parameter(model, k).detach().clone().cpu()
                    for k in weight_names}

    n_batches     = (len(dataset) + args.batch_size - 1) // args.batch_size
    all_evaluated     = []
    all_sublayer_attn = []
    all_sublayer_mlp  = []
    all_knockout      = []
    all_patch         = {source: [] for source in ["e1", "last"]}

    for b, start_idx in enumerate(range(0, len(dataset), args.batch_size)):
        batch_df = dataset.iloc[start_idx:start_idx + args.batch_size].copy()
        print(f"\n--- IFMET Batch {b+1}/{n_batches} ({len(batch_df)} examples) ---")

        # Edit — collect all requests for the batch, then call apply_algo once
        # per phase so the null-space projection covers all batch facts.
        standard_requests = []
        indirect_requests = []
        for _, row in batch_df.iterrows():
            standard_requests.append({"prompt": row["r1_template"], "subject": row["e1_label"], "target_new": row["e2_label"]})
            standard_requests.append({"prompt": row["r2_template"], "subject": row["e2_label"], "target_new": row["e3_label"]})
            indirect_requests.append({"prompt": row["source_prompt"], "subject": row["source_prompt"], "target_new": row["e3_label"]})
        hparams_s.batch_size = len(standard_requests)
        apply_algo(model, tokenizer, standard_requests, hparams_s, copy=False, return_orig_weights=False)
        hparams_d.batch_size = len(indirect_requests)
        apply_algo(model, tokenizer, indirect_requests, hparams_d, copy=False, return_orig_weights=False)
        print(f"  Batch edited ({len(standard_requests)} standard + {len(indirect_requests)} indirect requests).")

        # Evaluate
        print(f"  [Phase: Eval] Batch {b+1}/{n_batches}")
        fn_kwargs = {"model": model, "tokenizer": tokenizer}
        hf_b = Dataset.from_pandas(batch_df, preserve_index=False)
        hf_b = hf_b.map(eval_first_hop,         batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_second_hop,        batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_composition,       batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_entity_shortcut,   batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        hf_b = hf_b.map(eval_relation_shortcut, batched=True, batch_size=args.batch_size, fn_kwargs=fn_kwargs)
        all_evaluated.append(hf_b.to_pandas().drop(["e2_answers", "e3_answers"], axis=1).reset_index(drop=True))

        # Analyze
        if not args.skip_analysis:
            hf_analysis = Dataset.from_pandas(batch_df, preserve_index=False)
            print(f"  [Phase: Sublayer-Attn] Batch {b+1}/{n_batches}")
            all_sublayer_attn.append(to_sublayer_df(hf_analysis.map(
                project_attention, fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "attention"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names)))
            print(f"  [Phase: Sublayer-MLP] Batch {b+1}/{n_batches}")
            all_sublayer_mlp.append(to_sublayer_df(hf_analysis.map(
                project_attention, fn_kwargs={"model": model, "tokenizer": tokenizer, "target": "mlp"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names)))
            print(f"  [Phase: Knockout] Batch {b+1}/{n_batches}")
            all_knockout.append(hf_analysis.map(
                generate_answer_with_attention_knockout,
                fn_kwargs={"model": model, "tokenizer": tokenizer, "default_decoding": False,
                           "knockout_source": "last", "knockout_target": "e1", "k": 3, "prompt": "composition"},
                batched=True, batch_size=args.batch_size, remove_columns=hf_analysis.column_names).to_pandas())
            if not args.skip_patchscopes:
                print(f"  [Phase: PatchScopes] Batch {b+1}/{n_batches}")
                batch_ps = batch_df.copy()
                batch_ps["target_prompt"] = TARGET_PROMPT
                hf_ps = Dataset.from_pandas(batch_ps, preserve_index=False)
                for source in ["e1", "last"]:
                    all_patch[source].append(hf_ps.map(
                        generate_entity_description_with_patching,
                        fn_kwargs={"model": model, "tokenizer": tokenizer, "source": source, "do_sample": False},
                        batched=True, batch_size=args.batch_size, remove_columns=hf_ps.column_names).to_pandas())

        # Restore
        print(f"  [Phase: Restore] Batch {b+1}/{n_batches}")
        with torch.no_grad():
            for k, v in orig_weights.items():
                nethook.get_parameter(model, k)[...] = v.to(device)
        torch.cuda.empty_cache()
        print(f"  Weights restored to original.")

    evaluated_path = out_dir / "evaluated_two_hop.csv"
    evaluated = pd.concat(all_evaluated).reset_index(drop=True)
    evaluated.to_csv(evaluated_path)
    print(f"Saved {evaluated_path}")

    subsets = split_subsets(evaluated)
    for name, df in subsets.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / name / "two_hop.csv")
        print(f"Saved {out_dir / name / 'two_hop.csv'}")

    if not args.skip_analysis:
        analysis_subsets = subsets
        save_by_subset(pd.concat(all_sublayer_attn), analysis_subsets, out_dir,
                       "sublayer_projection", "attention_projections.csv", ["id", "layer"])
        save_by_subset(pd.concat(all_sublayer_mlp), analysis_subsets, out_dir,
                       "sublayer_projection", "mlp_projections.csv", ["id", "layer"])
        save_by_subset(pd.concat(all_knockout), analysis_subsets, out_dir,
                       "attention_knockout", "last_e1_attention_knockout_k3_composition.csv", ["id", "layer"])
        if not args.skip_patchscopes:
            for source in ["e1", "last"]:
                save_by_subset(pd.concat(all_patch[source]), analysis_subsets, out_dir,
                               "entity_description", f"{source}_entity_description_s0.csv",
                               ["id", "source_layer", "target_layer"])

    print(f"\nDone. Run: python analysis/analyze_tables.py --subsets "
          f"pre-edit:{data_dir}/composition_incorrect "
          f"now_correct:{out_dir}/now_correct "
          f"still_incorrect:{out_dir}/still_incorrect")


# %% Main

def main(args):
    for method in args.methods:

        print(f"\n{'='*70}")
        print(f"  METHOD: {method.upper()}")
        print(f"{'='*70}\n")

        if method == "preedit":
            run_preedit(args)
        elif method == "alphaedit":
            run_alphaedit(args)
        elif method == "ifmet":
            run_ifmet(args)
        elif method == "eap_alphaedit":
            run_eap_alphaedit(args)
        else:
            print(f"[SKIP] Unknown method: {method}")


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",    default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--methods",  nargs="+", default=["preedit", "alphaedit", "ifmet"],
                        choices=["preedit", "alphaedit", "ifmet", "eap_alphaedit"],
                        help="Which methods to run (default: all three)")
    parser.add_argument("--input",    default="datasets/two_hop.csv",
                        help="Raw two_hop.csv for pre-edit evaluation")
    parser.add_argument("--data-dir", default="datasets/meta-llama/Meta-Llama-3-8B",
                        help="Root data directory for this model")

    # AlphaEdit hparams
    parser.add_argument("--hparams-alphaedit",
                        default="EasyEdit/hparams/AlphaEdit/llama3-8b.yaml")

    # IFMET hparams
    parser.add_argument("--hparams-ifmet-shallow",
                        default="EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml")
    parser.add_argument("--hparams-ifmet-deep",
                        default="EasyEdit/hparams/IFMET/llama3-8b-deep.yaml")

    # Shared
    parser.add_argument("--max-examples",    type=int, default=None)
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip all analysis (sublayer, knockout, patchscopes) — edit + eval only")
    parser.add_argument("--skip-patchscopes", action="store_true",
                        help="Skip PatchScopes (method 4) to save time")
    parser.add_argument("--force", action="store_true",
                        help="Rerun and overwrite all existing outputs (ignore checkpoints and cached files)")
    parser.add_argument("--batch-size",        type=int, default=128,
                        help="Batch size for both inference (eval/analysis) and edit→analyze→restore cycles")
    parser.add_argument("--eap-top-k",        type=int, default=5,
                        help="Number of top MLP layers to select via EAP for eap_alphaedit")
    parser.add_argument("--eap-batch-size",   type=int, default=16,
                        help="Batch size for EAP attribution (smaller than batch-size due to gradient memory)")
    parser.add_argument("--eap-samples",      type=int, default=None,
                        help="Number of incorrect examples to run EAP on (default: all)")
    parser.add_argument("--device",          type=int, default=0)

    main(parser.parse_args())
