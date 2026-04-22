"""
Logit flow — cross-lingual analysis of how the edited knowledge builds up
layer by layer in vocabulary space.

Core question:
  AlphaEdit edits the model in English. At each full layer output, what is the
  probability that the model already "thinks" of the bridge entity (e2_new) or
  the final answer (e3_new)?

  Uses the logit lens (Nostalgebraist 2020): project the residual stream at the
  last token position through the model's final norm + unembedding at each layer.

  A spike in delta_e3 at layer L → that layer contributes most to assembling
  the final answer in vocabulary space.  Comparing EN vs. FR/DE/ZH/JA/HI:
    - EN peaks higher/earlier → edit is English-specific in representation space
    - All languages flat → edit didn't land in residual stream

Both bridge (e2_new) and final answer (e3_new) tokens are always English, same
as run_en_eval.py — eliminates cross-script tokenization bias.

Requires baukit:  pip install baukit

Usage:
    python analysis/run_logit_flow.py \\
        --model /workspace/models/ashwin/edited_model_qwen3-4b-instruct \\
        --dataset data/Qwen3-4B-Instruct-2507/two_hop_cf_e2_sampled_benchmark_multilingual.json \\
        --output_dir /workspace/output/logit_flow \\
        --row_role edit
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_layer_names, get_norm_module, get_prepend_space
from edit_utils import _format_chat_prompt


# ── constants ─────────────────────────────────────────────────────────────────

ALL_LANGUAGES = ["en", "fr", "de", "zh", "ja", "hi"]

LANG_SUFFIX = {
    "en": "", "fr": "_fr", "de": "_de",
    "zh": "_zh", "ja": "_ja", "hi": "_hi",
}


# ── dataset helpers ───────────────────────────────────────────────────────────

def _parse_aliases(val) -> list[str]:
    if val is None or (isinstance(val, float) and val != val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return [str(val)]


def _answer_list(label, aliases: list[str]) -> list[str]:
    out = []
    if label:
        out.append(str(label))
    out.extend(a for a in aliases if len(a) > 3 and a != label)
    return out


def _sample_id(sample: dict) -> int:
    return sample.get("id") or sample.get("case_id")


def _build_inputs(sample: dict, lang: str) -> dict | None:
    """
    Get the language-specific source prompt for this sample.

    bridge_answers (e2_new) is always English — it is what was edited.
    final_answers  (e3_new) is always English — eliminates cross-script
                            tokenization bias.
    Source prompt is language-specific — that is what changes across languages.
    """
    suf = LANG_SUFFIX[lang]
    lang_data = sample.get(lang) if isinstance(sample.get(lang), dict) else {}

    # source prompt in this language
    sp = (lang_data.get("source_prompt")
          or sample.get(f"source_prompt{suf}")
          or (sample.get("cloze_question") if lang == "en" else None))

    if not sp:
        return None

    # final answer — always English
    e3_new = sample.get("e3_label_new") or sample.get("new_answer")
    if e3_new is None or (isinstance(e3_new, float) and e3_new != e3_new):
        return None
    e3_aliases = _parse_aliases(
        sample.get("e3_aliases_new") or sample.get("new_answer_alias")
    )

    # bridge entity — always English
    e2_new = sample.get("e2_label_new")
    if e2_new is None:
        rr = sample.get("requested_rewrite", [])
        e2_new = rr[0].get("target_new", {}).get("str") if rr else None
    if e2_new is None:
        hops = sample.get("new_single_hops", [])
        e2_new = hops[0].get("answer") if hops else None
    if isinstance(e2_new, float) and e2_new != e2_new:
        e2_new = None

    e2_aliases = _parse_aliases(sample.get("e2_aliases_new"))

    return {
        "source_prompt":  sp,
        "bridge_answers": _answer_list(e2_new, e2_aliases),
        "final_answers":  _answer_list(e3_new, e3_aliases),
    }


# ── logit flow ────────────────────────────────────────────────────────────────

def _get_all_token_ids(tokenizer, answer_list: list[str], prepend_space: bool) -> list[list[int]]:
    """
    All token-ID sequences for all answer aliases (with/without leading space).

    Returns a list of token-ID lists, one per variant.  Preserves multi-token
    aliases so callers can average probability across tokens — avoiding the bias
    where a single common first token (e.g. Devanagari abbreviations, CJK
    characters) makes an alias look artificially likely.
    """
    all_seqs = []
    for ans in answer_list:
        for prefix in (("", " ") if prepend_space else ("",)):
            ids = tokenizer(prefix + ans, add_special_tokens=False).input_ids
            if ids:
                all_seqs.append(ids)
    return all_seqs


def _entity_prob(probs_layer: torch.Tensor, token_seqs: list[list[int]]) -> float:
    """
    Max mean-token-prob across all alias variants at a single layer.

    For each alias, take the mean probability across its tokens (each token
    evaluated independently at the last-token projection).  Then take the max
    across aliases.  This avoids first-token bias for non-Latin scripts and is
    consistent with the use_all_tokens=True mode in sublayer_projection.
    """
    if not token_seqs:
        return 0.0
    best = 0.0
    for toks in token_seqs:
        mean_p = probs_layer[toks].mean().item()
        if mean_p > best:
            best = mean_p
    return best


@torch.no_grad()
def _logit_flow_batch(model, tokenizer, prompts: list[str],
                      final_sets: list[list[str]],
                      bridge_sets: list[list[str]],
                      layer_names: list[str],
                      prepend_space: bool) -> list[dict]:
    """
    For each prompt × layer, compute the max mean-token probability among all
    answer alias variants when the full-layer residual stream is projected into
    vocabulary space (logit lens / Nostalgebraist 2020).

    Uses mean-across-tokens per alias → max across aliases to avoid first-token
    bias in non-Latin scripts (Devanagari, CJK).
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Capture full-layer residual stream at the last token position
    with TraceDict(model, layer_names) as trace:
        _ = model(**inputs).logits

    norm = get_norm_module(model)
    out_embed = model.get_output_embeddings()
    norm_device = next(norm.parameters()).device

    # hidden[i][l] = hidden state at layer l, last token, for prompt i
    hidden = []
    for i in range(len(prompts)):
        states = []
        for ln in layer_names:
            out = trace[ln].output
            if isinstance(out, tuple):
                out = out[0]
            states.append(out[i, -1].cpu())
        hidden.append(torch.stack(states))  # (num_layers, hidden_size)

    hidden_tensor = torch.stack(hidden).to(norm_device)  # (batch, layers, hidden)
    projections = out_embed(norm(hidden_tensor))          # (batch, layers, vocab)
    probs = F.softmax(projections.float(), dim=-1)        # float32 for stable softmax

    rows = []
    for i, (final_ans, bridge_ans) in enumerate(zip(final_sets, bridge_sets)):
        e3_seqs = _get_all_token_ids(tokenizer, final_ans, prepend_space)
        e2_seqs = _get_all_token_ids(tokenizer, bridge_ans, prepend_space)
        for l_idx in range(len(layer_names)):
            p = probs[i, l_idx]
            rows.append({
                "layer":   l_idx,
                "e3_prob": _entity_prob(p, e3_seqs),
                "e2_prob": _entity_prob(p, e2_seqs),
            })
    return rows


def run_logit_flow(model, tokenizer, dataset: list, languages: list,
                   batch_size: int) -> pd.DataFrame:
    """
    For every sample × language, run logit flow (full-layer residual stream
    projected into vocabulary space at the last token position).

    Output columns: id, row_role, lang, layer, e3_prob, e2_prob, delta_e3
      e3_prob:   probability of final answer (e3_new, English) at this layer
      e2_prob:   probability of bridge entity (e2_new, English) at this layer
      delta_e3:  e3_prob[layer] - e3_prob[layer-1]  (how much this layer adds)
    """
    try:
        prepend_space = get_prepend_space(model)
    except ValueError:
        prepend_space = True

    layer_names = get_layer_names(model)
    all_rows: list[dict] = []

    for lang in languages:
        print(f"\n  Language: {lang}")

        for i in tqdm(range(0, len(dataset), batch_size), desc=f"  Batches ({lang})"):
            batch = dataset[i: i + batch_size]
            prompts, final_sets, bridge_sets, ids, roles = [], [], [], [], []

            for s in batch:
                inp = _build_inputs(s, lang)
                if inp is None or not inp["bridge_answers"]:
                    continue
                prompts.append(_format_chat_prompt(inp["source_prompt"], tokenizer))
                final_sets.append(inp["final_answers"])
                bridge_sets.append(inp["bridge_answers"])
                ids.append(_sample_id(s))
                roles.append(s.get("row_role", "unknown"))

            if not prompts:
                continue

            try:
                layer_rows = _logit_flow_batch(
                    model, tokenizer, prompts, final_sets, bridge_sets,
                    layer_names, prepend_space,
                )
            except Exception as e:
                print(f"  [SKIP batch i={i} lang={lang}] {e}")
                continue

            # layer_rows is flat: len(prompts) × len(layer_names) entries
            num_layers = len(layer_names)
            for p_idx, (sid, role) in enumerate(zip(ids, roles)):
                for l_row in layer_rows[p_idx * num_layers: (p_idx + 1) * num_layers]:
                    all_rows.append({"id": sid, "row_role": role, "lang": lang, **l_row})

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["id", "lang", "layer"])
        df["delta_e3"] = df.groupby(["id", "lang"])["e3_prob"].diff().fillna(0.0)
    return df


# ── summary ───────────────────────────────────────────────────────────────────

def _print_logit_flow_block(df: pd.DataFrame, label: str, languages: list,
                             checkpoint_layers: list[int]):
    """
    Per-language logit-flow statistics for a subset of the DataFrame.

    Shows:
      - Mean e3_prob at selected checkpoint layers
      - Peak e3_prob layer and its value
      - Peak e2_prob value
      - First layer where mean e3_prob > 0.001  (where answer starts emerging)
    """
    print(f"\n--- {label} ---")
    if df.empty:
        print("  (no data)")
        return

    ckpt_hdr = "  ".join(f"L{l:02d}" for l in checkpoint_layers)
    print(f"  {'Lang':<6}  {ckpt_hdr}   {'peak_L':>6}  {'peak_e3':>8}  {'peak_e2':>8}  {'first>0.001':>11}")

    for lang in languages:
        sub = df[df["lang"] == lang]
        if sub.empty:
            continue
        by_layer_e3 = sub.groupby("layer")["e3_prob"].mean()
        by_layer_e2 = sub.groupby("layer")["e2_prob"].mean()

        ckpt_vals = "  ".join(
            f"{by_layer_e3.get(l, float('nan')):.4f}" for l in checkpoint_layers
        )
        peak_l   = int(by_layer_e3.idxmax())
        peak_e3  = float(by_layer_e3.max())
        peak_e2  = float(by_layer_e2.max())
        above    = by_layer_e3[by_layer_e3 > 0.001]
        first_above = int(above.index.min()) if not above.empty else -1

        print(f"  {lang:<6}  {ckpt_vals}   {peak_l:>6}  {peak_e3:>8.4f}  {peak_e2:>8.4f}  {first_above:>11}")


def _print_summary(df: pd.DataFrame, languages: list):
    num_layers = int(df["layer"].max()) + 1 if not df.empty else 36
    checkpoints = list(range(0, num_layers, 5))

    print("\n" + "=" * 80)
    print("LOGIT FLOW SUMMARY  (mean e3_prob / e2_prob across samples, per language)")
    print("=" * 80)
    print("e3_prob: probability mass on final-answer tokens at each layer (logit lens)")
    print("e2_prob: probability mass on bridge-entity tokens at each layer")
    print("Higher = model already 'thinks' of that entity at that layer.")
    print("EN peaking earlier/higher than others → edit is English-specific in repr space.")
    print("All languages flat → edit didn't land in residual stream.")
    print(f"\nCheckpoint layers shown: {checkpoints}")

    edit_df = df[df["row_role"] == "edit"]
    prop_df = df[df["row_role"] == "propagated_eval"]
    n_edit  = edit_df["id"].nunique()
    n_prop  = prop_df["id"].nunique()

    print(f"\n{'='*80}")
    print(f"row_role=edit  ({n_edit} samples)")
    print(f"{'='*80}")
    _print_logit_flow_block(edit_df, f"edit ({n_edit} samples)", languages, checkpoints)

    print(f"\n{'='*80}")
    print(f"row_role=propagated_eval  ({n_prop} samples)")
    print(f"{'='*80}")
    _print_logit_flow_block(prop_df, f"propagated_eval ({n_prop} samples)", languages, checkpoints)

    # Peak layer comparison across languages for edit rows
    if not edit_df.empty:
        print(f"\n--- Peak e3_prob layer per language (edit rows) ---")
        print(f"  Where EN peaks vs. other languages reveals edit transfer.")
        for lang in languages:
            sub = edit_df[edit_df["lang"] == lang]
            if sub.empty:
                continue
            by_layer = sub.groupby("layer")["e3_prob"].mean()
            peak_l  = int(by_layer.idxmax())
            peak_v  = float(by_layer.max())
            print(f"  {lang:<6}  peak at layer {peak_l:>3}  e3_prob={peak_v:.5f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.row_role:
        dataset = [s for s in dataset if s.get("row_role") == args.row_role]
        print(f"Filtered to row_role={args.row_role}: {len(dataset)} samples")

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    languages = args.languages
    print(f"Samples: {len(dataset)} | Languages: {languages} | Batch size: {args.batch_size}")

    print(f"\nLoading model from {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.float16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    df = run_logit_flow(model, tokenizer, dataset, languages, batch_size=args.batch_size)

    csv_path = os.path.join(args.output_dir, "logit_flow.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _print_summary(df, languages)
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--output_dir",  default="output/logit_flow")
    parser.add_argument("--languages",   nargs="+", default=ALL_LANGUAGES, choices=ALL_LANGUAGES)
    parser.add_argument("--row_role",    default=None, choices=["edit", "propagated_eval"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=4)
    main(parser.parse_args())
