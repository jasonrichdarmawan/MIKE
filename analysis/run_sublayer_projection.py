"""
Cross-lingual sublayer projection 

Core question:
  AlphaEdit edits the model in English. For the same sample, when the question
  is asked in French / German / Chinese / Japanese / Hindi, does the edited
  bridge entity (e2_new) still appear in the residual stream?

  If bridge rank is low in English but high in other languages → the edit is
  English-specific in representation space and did not transfer.
  If bridge rank is similarly low across languages → the edit transferred.

No behavioral eval. No filtering by correct/incorrect.
Just sublayer projection across all samples × all languages.

Usage:
    python analysis/run_en_eval.py \\
        --model /workspace/models/ashwin/edited_model_qwen3-4b-instruct \\
        --dataset data/Qwen3-4B-Instruct-2507/two_hop_cf_e2_sampled_benchmark_multilingual.json \\
        --output_dir /workspace/output/crosslingual_proj \\
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from patched_generation import get_attention_projection
from edit_utils import _format_chat_prompt


# ── constants ─────────────────────────────────────────────────────────────────

ALL_LANGUAGES = ["en", "fr", "de", "zh", "ja", "hi"]

LANG_SUFFIX = {
    "en": "", "fr": "_fr", "de": "_de",
    "zh": "_zh", "ja": "_ja", "hi": "_hi",
}

LATE_LAYERS = range(27, 36)


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
                            tokenization bias (same principle as bridge).
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


# ── sublayer projection ───────────────────────────────────────────────────────

def run_projection(model, tokenizer, dataset: list, languages: list,
                   log_file=None) -> pd.DataFrame:
    """
    For every sample × language, run sublayer projection (attention sublayers).
    Records bridge_rank_min and answer_rank_min per layer.

    bridge_answers = English e2_new (same across all languages)
    final_answers  = language-specific e3_new
    """
    from project_sublayer import project_attention

    all_rows: list[dict] = []

    def _log(msg):
        if log_file:
            log_file.write(msg + "\n")

    for lang in languages:
        skipped = 0
        print(f"\n  Language: {lang}")
        _log(f"\n{'='*80}\nLANGUAGE: {lang.upper()}\n{'='*80}")

        for s in tqdm(dataset, desc=f"  {lang}"):
            sid  = _sample_id(s)
            role = s.get("row_role", "edit")
            inp  = _build_inputs(s, lang)

            if inp is None:
                skipped += 1
                continue

            if not inp["bridge_answers"]:
                skipped += 1
                continue

            prompt = _format_chat_prompt(inp["source_prompt"], tokenizer)

            entries = {
                "id":            [sid],
                "source_prompt": [prompt],
                "e2_answers":    [inp["bridge_answers"]],
                "e3_answers":    [inp["final_answers"]],
            }

            try:
                result = project_attention(entries, model, tokenizer, target="attention")
            except Exception as e:
                _log(f"[SKIP id={sid} lang={lang}] {e}")
                skipped += 1
                continue

            for i, layer in enumerate(result["layer"]):
                e2_r = result["e2_ranks"][i]
                e3_r = result["e3_ranks"][i]
                all_rows.append({
                    "id":              sid,
                    "row_role":        role,
                    "lang":            lang,
                    "layer":           layer,
                    "bridge_rank_min": int(min(e2_r)) if e2_r else 999999,
                    "answer_rank_min": int(min(e3_r)) if e3_r else 999999,
                })

        print(f"    skipped: {skipped}")

    return pd.DataFrame(all_rows)


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, languages: list):
    print("\n" + "=" * 80)
    print("CROSS-LINGUAL SUBLAYER PROJECTION SUMMARY  (attention sublayers)")
    print("=" * 80)
    print("bridge = e2_new (English-edited entity) | answer = e3_new (language-specific)")
    print("Lower rank = entity appears higher in residual stream projection.")
    print("EN bridge rank is the baseline — other languages vs EN shows transfer.")
    print()

    late = df[df["layer"].isin(LATE_LAYERS)]

    print(f"  {'Lang':<6}  {'N samples':>10}  "
          f"{'bridge mean L27-35':>18}  {'bridge ≤200':>11}  {'bridge ≤1000':>12}  "
          f"{'answer(EN) mean':>15}  {'answer(EN) ≤200':>15}")
    print(f"  {'':6}  {'':10}  {'':18}  {'':11}  {'':12}  "
          f"  (answer always English — no tokenization bias)")

    for lang in languages:
        sub      = df[df["lang"] == lang]
        late_sub = late[late["lang"] == lang]
        if sub.empty:
            print(f"  {lang:<6}  {'0':>10}")
            continue

        n     = sub["id"].nunique()
        bm    = late_sub["bridge_rank_min"].mean()   if not late_sub.empty else float("nan")
        b200  = int((late_sub["bridge_rank_min"] <= 200).sum())
        b1000 = int((late_sub["bridge_rank_min"] <= 1000).sum())
        am    = late_sub["answer_rank_min"].mean()   if not late_sub.empty else float("nan")
        a200  = int((late_sub["answer_rank_min"] <= 200).sum())

        print(f"  {lang:<6}  {n:>10,}  "
              f"{bm:>18.1f}  {b200:>11,}  {b1000:>12,}  "
              f"{am:>15.1f}  {a200:>15,}")

    # Per-sample detail: for EN, show samples where bridge rank was low
    print()
    print(f"--- EN samples with bridge rank ≤ 200 in late layers ---")
    en_late = late[late["lang"] == "en"]
    low_bridge = en_late[en_late["bridge_rank_min"] <= 200]["id"].unique()
    print(f"  {len(low_bridge)} sample(s): {sorted(low_bridge)}")

    if len(low_bridge) > 0:
        print()
        print(f"--- Same samples: bridge rank per language (mean L27-35) ---")
        print(f"  {'id':<8}" + "".join(f"  {lang:>8}" for lang in languages))
        for sid in sorted(low_bridge):
            row = f"  {sid:<8}"
            for lang in languages:
                sub = late[(late["id"] == sid) & (late["lang"] == lang)]
                val = sub["bridge_rank_min"].mean() if not sub.empty else float("nan")
                row += f"  {val:>8.0f}"
            print(row)

        print()
        print(f"--- Same samples: answer(EN) rank per language (mean L27-35) ---")
        print(f"  {'id':<8}" + "".join(f"  {lang:>8}" for lang in languages))
        for sid in sorted(low_bridge):
            row = f"  {sid:<8}"
            for lang in languages:
                sub = late[(late["id"] == sid) & (late["lang"] == lang)]
                val = sub["answer_rank_min"].mean() if not sub.empty else float("nan")
                row += f"  {val:>8.0f}"
            print(row)


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
    print(f"Samples: {len(dataset)} | Languages: {languages}")

    print(f"\nLoading model from {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.float16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "sublayer_diag.log")
    with open(log_path, "w", encoding="utf-8") as lf:
        proj_df = run_projection(model, tokenizer, dataset, languages, log_file=lf)

    csv_path = os.path.join(args.output_dir, "sublayer_proj.csv")
    proj_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _print_summary(proj_df, languages)
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output_dir", default="output/crosslingual_proj")
    parser.add_argument("--languages",  nargs="+", default=ALL_LANGUAGES,
                        choices=ALL_LANGUAGES)
    parser.add_argument("--row_role",   default=None, choices=["edit", "propagated_eval"])
    parser.add_argument("--max_samples", type=int, default=None)
    main(parser.parse_args())
