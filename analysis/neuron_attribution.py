"""
Compute neuron-level LPI and chained backward attribution for all examples.

Replaces neuron_lpi.py + neuron_query.py: loads the model once, loops over all
subsets, and for each example runs all requested prompt types together (sharing
one tokeniser/model call per prompt) so cross-prompt Jaccard can be computed
on-the-fly without saving large per-example CSV files.

Outputs per (prompt_type, subset) under <data_dir>/neuron_attribution/<subset>/<prompt_type>/:
  sublayer_lpi.csv        — per-(example, layer, sublayer): lpi + top_k_tokens
  head_lpi.csv            — per-(example, layer, head):     lpi + top_k_tokens
  ffn_query_layers.csv    — LPI-weighted residual contribution of prior layers to top-K FFN neurons
  attn_query_layers.csv   — LPI-weighted residual contribution of prior layers to top-K attn neurons
  neuron_layer_dist.csv   — average neuron count per layer per example (all orders and ntypes)
  chain_profile.csv       — mean attribution score per (order, ntype, layer)
                            ntypes: "ffn"/"attn" at order 0; "attn→ffn"/"ffn→attn" at order -1;
                            "ffn→attn" only at order -2
  chain_mean_layer.csv    — score-weighted mean layer per (order, ntype)

Outputs per subset under <data_dir>/neuron_attribution/<subset>/ (covering all prompt-type pairs):
  jaccard.csv             — mean Jaccard similarity per (pair, neuron_class, ntype_a, order_a,
                            ntype_b, order_b, layer).  neuron_class is "ffn" or "attn" and
                            groups same-class neurons regardless of chain ntype.
                            Same-order same-ntype rows: used for value/query neuron overlap plots.
                            Cross-order or cross-ntype (but same neuron_class) rows: used for
                            the +1-order-shift chain-hop heatmaps.

Example output tree for --data-dir datasets/gpt2:
    datasets/gpt2/neuron_attribution/
        correct/
            two-hop/          sublayer_lpi.csv, chain_profile.csv, ...
            first-hop/        sublayer_lpi.csv, ...
            second-hop/       sublayer_lpi.csv, ...
            jaccard.csv
        incorrect/
            ...
        inconsistent/
            ...

Usage:
    python analysis/neuron_attribution.py
    python analysis/neuron_attribution.py --model-name gpt2 --subsets correct incorrect
    python analysis/neuron_attribution.py --prompts two-hop first-hop second-hop
"""

import argparse
import math
import pickle
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import patch_model
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Prompt type names are used directly as subdirectory names under
# <data_dir>/neuron_attribution/<subset>/<prompt_type>/
PROMPT_TYPES = ["two-hop", "first-hop", "second-hop"]


# ── Projection helpers ────────────────────────────────────────────────────────

def project_to_vocab(vectors, final_std, ln_f_weight, lm_head):
    """
    Approximate final-LayerNorm + lm_head projection.

    vectors: (..., hidden_dim)
    returns: (..., vocab_size) logits
    """
    E = vectors.mean(-1, keepdim=True)
    normed = (vectors - E) / final_std * ln_f_weight
    return lm_head(normed)


def decode_top_k(vectors, final_std, ln_f_weight, lm_head, tokenizer, k=10):
    """
    Project vectors through the unembedding and return the top-k token strings.

    vectors: (..., hidden_dim)
    returns: list of top-k token strings per vector
    """
    logits = project_to_vocab(vectors, final_std, ln_f_weight, lm_head)
    top_idx = logits.topk(k, dim=-1).indices
    if top_idx.dim() == 1:
        return tokenizer.convert_ids_to_tokens(top_idx.tolist())
    flat = top_idx.reshape(-1, k).tolist()
    return [tokenizer.convert_ids_to_tokens(row) for row in flat]


# ── LPI computations ──────────────────────────────────────────────────────────

def compute_layer_lpi(states, layer_num, predict_index, final_std, ln_f_weight, lm_head,
                      tokenizer, top_k_proj=10):
    """
    Layer-level LPI: how much does each attention / FFN sublayer increase the
    log-probability of the predicted token?

    Returns: list of (layer_i, sublayer, lpi, top_k_tokens)
    """
    rows = []
    for i in range(layer_num):
        layer_input_last = states[i]["layer_input"][-1]
        attn_output_last = states[i]["attn_output"][-1]
        residual_last    = states[i]["residual_output"][-1]
        ffn_output_last  = states[i]["ffn_output"][-1]

        base  = torch.log_softmax(project_to_vocab(layer_input_last, final_std, ln_f_weight, lm_head), -1)[predict_index]
        after = torch.log_softmax(project_to_vocab(layer_input_last + attn_output_last, final_std, ln_f_weight, lm_head), -1)[predict_index]
        attn_tokens = decode_top_k(attn_output_last, final_std, ln_f_weight, lm_head, tokenizer, k=top_k_proj)
        rows.append((i, "attention", (after - base).item(), attn_tokens))

        base  = torch.log_softmax(project_to_vocab(residual_last, final_std, ln_f_weight, lm_head), -1)[predict_index]
        after = torch.log_softmax(project_to_vocab(residual_last + ffn_output_last, final_std, ln_f_weight, lm_head), -1)[predict_index]
        ffn_tokens = decode_top_k(ffn_output_last, final_std, ln_f_weight, lm_head, tokenizer, k=top_k_proj)
        rows.append((i, "ffn", (after - base).item(), ffn_tokens))
    return rows


def compute_head_lpi(states, info, layer_num, head_num, head_dim,
                     predict_index, final_std, ln_f_weight, lm_head,
                     tokenizer, top_k_proj=10):
    """
    Head-level LPI: per-(layer, head) attention contribution.

    Returns list of (layer_i, head_i, lpi_score, top_k_tokens).
    """
    rows = []
    for layer_i in range(layer_num):
        layer_input_last = states[layer_i]["layer_input"][-1]
        v_heads = states[layer_i]["attn_subvalues"]
        o_proj  = info["get_attn_o_proj"](info["layers"][layer_i])
        head_sum = torch.bmm(v_heads, o_proj).sum(1)

        base = torch.log_softmax(project_to_vocab(layer_input_last, final_std, ln_f_weight, lm_head), -1)[predict_index]
        head_plus = head_sum + layer_input_last
        logits = project_to_vocab(head_plus, final_std, ln_f_weight, lm_head)
        lpi = torch.log_softmax(logits, -1)[:, predict_index] - base
        head_tokens = decode_top_k(head_sum, final_std, ln_f_weight, lm_head, tokenizer, k=top_k_proj)

        for head_i, score in enumerate(lpi.tolist()):
            rows.append((layer_i, head_i, score, head_tokens[head_i]))
    return rows


def compute_ffn_neuron_lpi(states, info, layer_num, predict_index,
                            final_std, ln_f_weight, lm_head, top_k=300):
    """
    FFN value neuron LPI: per-neuron contribution to the predicted token.

    Returns top_k (layer_i, neuron_id, pos, lpi_score) tuples globally,
    sorted descending by score. pos is always -1 (last token).
    """
    all_neurons = []
    for layer_i in range(layer_num):
        coeff    = states[layer_i]["coeff_scores"][-1]
        fc2      = info["get_ffn_down_proj"](info["layers"][layer_i])
        residual = states[layer_i]["residual_output"][-1]
        neuron_contribs = coeff.unsqueeze(-1) * fc2

        base = torch.log_softmax(project_to_vocab(residual, final_std, ln_f_weight, lm_head), -1)[predict_index]
        plus   = neuron_contribs + residual
        logits = project_to_vocab(plus, final_std, ln_f_weight, lm_head)
        lpi    = torch.log_softmax(logits, -1)[:, predict_index] - base

        k = min(top_k, lpi.shape[0])
        topk_scores, topk_idx = lpi.topk(k)
        for neuron_id, score in zip(topk_idx.tolist(), topk_scores.tolist()):
            all_neurons.append((layer_i, neuron_id, -1, score))

    all_neurons.sort(key=lambda x: x[3], reverse=True)
    return all_neurons[:top_k]


def compute_attn_neuron_lpi(states, info, layer_num, head_num, head_dim,
                             predict_index, final_std, ln_f_weight, lm_head, top_k=300):
    """
    Attention value neuron LPI: per-(layer, head, neuron_dim, source_pos) contribution.

    Processes one source position at a time to bound peak memory at
    (head_num × head_dim × hidden_dim) + (head_num × head_dim × vocab_size) per step,
    avoiding the full (seq_len × head_num × head_dim × vocab_size) intermediate tensor.

    Returns top_k (layer_i, head_i, neuron_id, pos, lpi_score) tuples globally,
    sorted descending by score.
    """
    all_neurons = []
    for layer_i in range(layer_num):
        layer_input_last = states[layer_i]["layer_input"][-1]
        attn_sv = states[layer_i]["attn_subvalues"]           # (head_num, seq_len, head_dim)
        o_proj  = info["get_attn_o_proj"](info["layers"][layer_i])  # (head_num, head_dim, hidden_dim)
        seq_len = attn_sv.shape[1]

        base = torch.log_softmax(
            project_to_vocab(layer_input_last, final_std, ln_f_weight, lm_head), -1
        )[predict_index]

        layer_neurons = []
        for pos in range(seq_len):
            v_pos = attn_sv[:, pos, :]                        # (head_num, head_dim)
            neuron_contribs = v_pos.unsqueeze(-1) * o_proj    # (head_num, head_dim, hidden_dim)
            plus   = neuron_contribs + layer_input_last
            logits = project_to_vocab(plus, final_std, ln_f_weight, lm_head)
            lpi    = torch.log_softmax(logits, -1)[..., predict_index] - base  # (head_num, head_dim)

            k_pos = min(top_k, lpi.numel())
            topk_scores, topk_flat_idx = lpi.reshape(-1).topk(k_pos)
            for flat_i, score in zip(topk_flat_idx.tolist(), topk_scores.tolist()):
                head_i   = flat_i // head_dim
                neuron_i = flat_i % head_dim
                layer_neurons.append((layer_i, head_i, neuron_i, pos, score))

        layer_neurons.sort(key=lambda x: x[4], reverse=True)
        all_neurons.extend(layer_neurons[:top_k])

    all_neurons.sort(key=lambda x: x[4], reverse=True)
    return all_neurons[:top_k]


# ── Query computations ────────────────────────────────────────────────────────

def _build_residual_components(states, layer_limit, pos):
    """
    Collect residual stream components at token position `pos` up to `layer_limit`.
    """
    components = [states[0]["layer_input"][pos]]
    for i in range(layer_limit):
        components.append(states[i]["attn_output"][pos])
        components.append(states[i]["ffn_output"][pos])
    return components


def compute_ffn_query_layers(states, info, top_neurons, layer_num):
    """
    Residual stream decomposition for top-K FFN value neurons.

    Returns: list of (layer_i, sublayer, score)
    """
    stacked_cache = {}  # ffn_layer -> stacked residual tensor (reused across neurons)
    all_scores = [0.0] * (1 + 2 * layer_num)
    for ffn_layer, neuron_id, _pos, lpi_score in top_neurons:
        key_scaled = (
            info["get_ffn_key"](info["layers"][ffn_layer], neuron_id)
            * info["get_ffn_ln_weight"](info["layers"][ffn_layer])
        )
        if ffn_layer not in stacked_cache:
            components = _build_residual_components(states, ffn_layer, -1)
            components.append(states[ffn_layer]["attn_output"][-1])
            stacked_cache[ffn_layer] = torch.stack(components)
        stacked = stacked_cache[ffn_layer]
        ips = (stacked * key_scaled).sum(-1)
        total = ips.sum().item()
        if abs(total) < 1e-9:
            continue
        for idx, ip in enumerate(ips.tolist()):
            all_scores[idx] += (ip / total) * lpi_score

    rows = []
    for idx in range(1, len(all_scores)):
        layer_i  = (idx - 1) // 2
        sublayer = "attention" if (idx - 1) % 2 == 0 else "ffn"
        rows.append((layer_i, sublayer, all_scores[idx]))
    return rows


def compute_attn_query_layers(states, info, top_neurons, layer_num, head_dim,
                              head_num, num_kv_heads):
    """
    Residual stream decomposition for top-K attention value neurons.

    Returns: list of (layer_i, sublayer, score)
    """
    num_groups  = head_num // num_kv_heads
    stacked_cache = {}  # (attn_layer, pos) -> stacked residual tensor
    all_scores = [0.0] * (1 + 2 * layer_num)
    for attn_layer, head_i, neuron_i, pos, lpi_score in top_neurons:
        kv_head_i   = head_i // num_groups
        flat_neuron = kv_head_i * head_dim + neuron_i
        key_scaled = (
            info["get_attn_v_key"](info["layers"][attn_layer], flat_neuron)
            * info["get_attn_ln_weight"](info["layers"][attn_layer])
        )
        cache_key = (attn_layer, pos)
        if cache_key not in stacked_cache:
            stacked_cache[cache_key] = torch.stack(
                _build_residual_components(states, attn_layer, pos)
            )
        stacked = stacked_cache[cache_key]
        ips = (stacked * key_scaled).sum(-1)
        total = ips.sum().item()
        if abs(total) < 1e-9:
            continue
        for idx, ip in enumerate(ips.tolist()):
            all_scores[idx] += (ip / total) * lpi_score

    rows = []
    for idx in range(1, len(all_scores)):
        layer_i  = (idx - 1) // 2
        sublayer = "attention" if (idx - 1) % 2 == 0 else "ffn"
        rows.append((layer_i, sublayer, all_scores[idx]))
    return rows


def compute_attn_query_neurons(states, info, top_neurons, head_dim,
                               head_num, num_kv_heads, coeff_stack, top_k=300):
    """
    Find which FFN neurons from prior layers most activate the top-K attn value neurons.

    Tracks each found FFN neuron's source position (the sequence position at which the
    FFN fired, i.e. the same position the attn V-key reads from).

    coeff_stack: pre-stacked coeff_scores (layer_num, seq_len, ffn_dim) from process_example.
    info must contain 'fc2_stack' (layer_num, ffn_dim, hidden_dim), pre-stacked in main().

    Returns: list of (layer_i, neuron_id, pos, aggregate_score)
    """
    num_groups = head_num // num_kv_heads
    layer_num  = len(states)
    ffn_dim    = coeff_stack.shape[-1]
    seq_len    = coeff_stack.shape[1]
    device     = coeff_stack.device

    # acc[layer, pos, neuron] — per source-position to preserve positional identity
    acc = torch.zeros(layer_num, seq_len, ffn_dim, device=device)

    for attn_layer, head_i, neuron_i, pos, lpi_score in top_neurons:
        kv_head_i   = head_i // num_groups
        flat_neuron = kv_head_i * head_dim + neuron_i
        key_scaled = (
            info["get_attn_v_key"](info["layers"][attn_layer], flat_neuron)
            * info["get_attn_ln_weight"](info["layers"][attn_layer])
        )
        total = (states[attn_layer]["layer_input"][pos] * key_scaled).sum().item()
        if abs(total) < 1e-9:
            continue
        scale = lpi_score / total
        if attn_layer > 0:
            # Vectorised over layers: (attn_layer, ffn_dim, hidden_dim) @ (hidden_dim,) -> (attn_layer, ffn_dim)
            fc2_proj = info["fc2_stack"][:attn_layer] @ key_scaled
            acc[:attn_layer, pos] += fc2_proj * coeff_stack[:attn_layer, pos] * scale

    flat = acc.reshape(-1)
    k = min(top_k, flat.numel())
    topk_vals, topk_idx = flat.topk(k)
    stride_l = seq_len * ffn_dim
    result = []
    for flat_i, score in zip(topk_idx.tolist(), topk_vals.tolist()):
        layer_i   = flat_i // stride_l
        rem       = flat_i % stride_l
        pos_out   = rem // ffn_dim
        neuron_id = rem % ffn_dim
        # Normalize last position to -1 for consistency with order-0 FFN tuples
        if pos_out == seq_len - 1:
            pos_out = -1
        result.append((layer_i, neuron_id, pos_out, score))
    return result


def compute_ffn_query_neurons(states, info, top_neurons, head_num, head_dim,
                              attn_sv_stack, top_k=300):
    """
    Find which attention value neurons from prior layers most activate the top-K FFN neurons.

    top_neurons: list of (layer_i, neuron_id, pos, lpi_score)
      pos == -1 means last token (order-0 FFN neurons).
      pos != -1 means a non-last source position (order≥1 chain neurons from
      compute_attn_query_neurons); attn_subvalues only stores last-token attention
      patterns, so the attn-source breakdown is skipped for those — only last-position
      FFN targets contribute to the output.

    Also fixes: same-layer attention (layer_i == ffn_layer) is now included,
    since it writes directly into the FFN's input residual.

    attn_sv_stack: pre-stacked attn_subvalues (layer_num, head_num, seq_len, head_dim).
    info must contain 'o_proj_stack' (layer_num, head_num, head_dim, hidden_dim).

    Returns: list of (layer_i, head_i, neuron_id, pos, aggregate_score)
    """
    layer_num = len(states)
    seq_len   = attn_sv_stack.shape[2]
    device    = attn_sv_stack.device

    acc = torch.zeros(layer_num, head_num, seq_len, head_dim, device=device)

    for ffn_layer, neuron_id, pos, lpi_score in top_neurons:
        key_scaled = (
            info["get_ffn_key"](info["layers"][ffn_layer], neuron_id)
            * info["get_ffn_ln_weight"](info["layers"][ffn_layer])
        )
        residual = states[ffn_layer]["residual_output"][pos]
        total = (residual * key_scaled).sum().item()
        if abs(total) < 1e-9:
            continue
        scale = lpi_score / total

        # attn_subvalues = alpha[last_token, src_pos] * V[src_pos] — valid only for
        # last-position FFN targets. Skip non-last positions to avoid wrong attribution.
        if pos != -1:
            continue

        # Vectorised over layers: (ffn_layer+1, head_num, head_dim, hidden_dim) @ (hidden_dim,)
        # -> (ffn_layer+1, head_num, head_dim)
        dot_all = info["o_proj_stack"][:ffn_layer + 1] @ key_scaled
        # attn_sv_stack[:ffn_layer+1]: (ffn_layer+1, head_num, seq_len, head_dim)
        # dot_all.unsqueeze(2):        (ffn_layer+1, head_num, 1,        head_dim)
        acc[:ffn_layer + 1] += attn_sv_stack[:ffn_layer + 1] * dot_all.unsqueeze(2) * scale

    flat = acc.reshape(-1)
    k = min(top_k, flat.numel())
    topk_vals, topk_idx = flat.topk(k)
    stride_h = seq_len * head_dim
    stride_l = head_num * stride_h
    result = []
    for flat_i, score in zip(topk_idx.tolist(), topk_vals.tolist()):
        layer_i  = flat_i // stride_l
        rem      = flat_i % stride_l
        head_i   = rem // stride_h
        rem2     = rem % stride_h
        pos      = rem2 // head_dim
        neuron_i = rem2 % head_dim
        result.append((layer_i, head_i, neuron_i, pos, score))
    return result


# ── Per-example driver ────────────────────────────────────────────────────────

def process_example(model, info, tokenizer, prompt, layer_num, head_num, head_dim,
                    num_kv_heads, top_k_ffn, top_k_proj, top_k_query, max_order=2):
    """
    Run one forward pass and compute all LPI and fixed 2-order backward attribution.

    Chain structure (max_order controls depth, orders are negative = backward from prediction):
        order  0: "ffn"      — FFN value neurons (LPI)
                  "attn"     — attn value neurons (LPI)
        order -1: "attn→ffn" — attn neurons activating order-0 FFN  (FFN key decomp)
                  "ffn→attn" — FFN neurons activating order-0 attn  (attn V-key decomp)
        order -2: "ffn→attn" — FFN neurons activating order-(-1) attn  (attn V-key decomp)
                  ("attn→ffn" omitted at order -2: order-(-1) FFN are at non-last positions,
                   attn_subvalues unavailable for those)
    max_order=2 for two-hop; max_order=1 for first-hop and second-hop.

    Returns:
        layer_rows   — [(layer_i, sublayer, lpi, top_k_tokens), ...]
        head_rows    — [(layer_i, head_i, lpi, top_k_tokens), ...]
        ffn_top      — [(layer_i, neuron_id, pos, score), ...]   (order-0 FFN, pos=-1)
        attn_top     — [(layer_i, head_i, neuron_i, pos, score), ...]
        ffn_query_rows  — [(layer_i, sublayer, score), ...]
        attn_query_rows — [(layer_i, sublayer, score), ...]
        chain        — [(order, ntype, rows), ...]
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor([tokens], device=device)

    with torch.no_grad():
        model(tokens_tensor, use_cache=False)

    states = patch_model.extract_states(layer_num)

    with torch.no_grad():
        last_hidden = states[-1]["layer_output"][-1]
        final_std = (last_hidden.var(unbiased=False) + 1e-5).sqrt().item()
        predict_index = info["lm_head"](
            (last_hidden - last_hidden.mean()) / final_std * info["ln_f_weight"]
        ).argmax().item()

        ln_f_weight = info["ln_f_weight"]
        lm_head     = info["lm_head"]

        # Pre-stack per-example tensors once; avoids repeated dict-access inside loops
        coeff_stack   = torch.stack([s["coeff_scores"]   for s in states])  # (L, seq, ffn_dim)
        attn_sv_stack = torch.stack([s["attn_subvalues"] for s in states])  # (L, head, seq, d)

        layer_rows = compute_layer_lpi(
            states, layer_num, predict_index, final_std, ln_f_weight, lm_head,
            tokenizer, top_k_proj=top_k_proj)
        head_rows = compute_head_lpi(
            states, info, layer_num, head_num, head_dim,
            predict_index, final_std, ln_f_weight, lm_head,
            tokenizer, top_k_proj=top_k_proj)
        ffn_top = compute_ffn_neuron_lpi(
            states, info, layer_num, predict_index,
            final_std, ln_f_weight, lm_head, top_k=top_k_ffn)
        attn_top = compute_attn_neuron_lpi(
            states, info, layer_num, head_num, head_dim,
            predict_index, final_std, ln_f_weight, lm_head, top_k=top_k_ffn)

        ffn_query_rows  = compute_ffn_query_layers(states, info, ffn_top, layer_num)
        attn_query_rows = compute_attn_query_layers(states, info, attn_top, layer_num, head_dim,
                                                    head_num, num_kv_heads)

        chain = []

        # Order 1
        ord1_attn = compute_ffn_query_neurons(
            states, info, ffn_top, head_num, head_dim, attn_sv_stack, top_k=top_k_query)
        ord1_ffn  = compute_attn_query_neurons(
            states, info, attn_top, head_dim, head_num, num_kv_heads, coeff_stack,
            top_k=top_k_query)
        if ord1_attn:
            chain.append((-1, "attn→ffn", ord1_attn))
        if ord1_ffn:
            chain.append((-1, "ffn→attn", ord1_ffn))

        # Order -2: only "ffn→attn" from order-(-1) "attn→ffn"
        # "attn→ffn" at order -2 is omitted — order-(-1) FFN are at non-last positions
        # and attn_subvalues does not cover per-position attention patterns.
        if max_order >= 2 and ord1_attn:
            ord2_ffn = compute_attn_query_neurons(
                states, info, ord1_attn, head_dim, head_num, num_kv_heads, coeff_stack,
                top_k=top_k_query)
            if ord2_ffn:
                chain.append((-2, "ffn→attn", ord2_ffn))

    return (layer_rows, head_rows, ffn_top, attn_top,
            ffn_query_rows, attn_query_rows, chain)


# ── Neuron-set helpers for Jaccard ────────────────────────────────────────────

def _get_prompt(row, prompt_type):
    if prompt_type == "two-hop":
        return row["source_prompt"]
    elif prompt_type == "first-hop":
        return f"{row['r1_template'].format(row['e1_label'])} is"
    elif prompt_type == "second-hop":
        return f"{row['r2_template'].format(row['e2_label'])} is"


def _get_position_labels(tokens, row, tokenizer, prompt_type):
    """
    Map each token position in `tokens` to a semantic label.

    Two-hop prompt structure: [r2 tokens][r1 tokens][e1 tokens][last]
        "e1"   — explicit entity tokens (e1_label)
        "r1"   — inner relation tokens (r1_template minus entity placeholder)
        "r2"   — outer relation tokens (r2_template minus entity placeholder)
        "last" — final token (prediction position)

    First-hop: [r1 tokens][e1 tokens][last]  → labels: e1, r1, last
    Second-hop: [r2 tokens][e2_label tokens][last]  → labels: e1 (e2_label in
        entity position), r2, last  (e2 is the entity for this hop)

    Tries with/without leading space to handle GPT-2/Llama tokenization.
    Unmatched tokens fall back to "other".
    """
    n = len(tokens)
    labels = ["other"] * n
    if n > 0:
        labels[-1] = "last"

    e1 = str(row.get("e1_label") or "")
    e2 = str(row.get("e2_label") or "")
    r1_tmpl = str(row.get("r1_template") or "")
    r2_tmpl = str(row.get("r2_template") or "")

    def _find_span(text):
        if not text:
            return None
        for prefix in [" " + text, text]:
            sub = tokenizer.encode(prefix, add_special_tokens=False)
            if not sub:
                continue
            for start in range(n - len(sub) + 1):
                if tokens[start : start + len(sub)] == sub:
                    return (start, start + len(sub))
        return None

    def _label_span(text, label):
        span = _find_span(text)
        if span:
            for i in range(span[0], span[1]):
                if labels[i] == "other":
                    labels[i] = label

    if prompt_type == "two-hop":
        # Mark e1 first (highest priority)
        _label_span(e1, "e1")
        # Mark full r1 clause (r1_template.format(e1)), then non-e1 tokens within it → r1
        if r1_tmpl and e1:
            _label_span(r1_tmpl.format(e1), "r1")
        # Remaining unlabeled tokens (excluding last) belong to r2
        for i in range(n - 1):
            if labels[i] == "other":
                labels[i] = "r2"

    elif prompt_type == "first-hop":
        _label_span(e1, "e1")
        # All remaining non-last tokens are r1
        for i in range(n - 1):
            if labels[i] == "other":
                labels[i] = "r1"

    elif prompt_type == "second-hop":
        # e2_label occupies the entity position in this single-hop prompt
        _label_span(e2, "e1")
        # All remaining non-last tokens are r2
        for i in range(n - 1):
            if labels[i] == "other":
                labels[i] = "r2"

    return labels


def _ffn_sets_by_layer(neurons):
    """[(layer_i, neuron_id, pos, score), ...] -> {layer: frozenset(neuron_id)}"""
    d = defaultdict(set)
    for layer_i, neuron_id, _pos, score in neurons:
        d[layer_i].add(neuron_id)
    return {l: frozenset(s) for l, s in d.items()}


def _ffn_pos_by_layer(neurons):
    """[(layer_i, neuron_id, pos, score), ...] -> {layer: {neuron_id: [pos, ...]}}"""
    d = defaultdict(lambda: defaultdict(list))
    for layer_i, neuron_id, pos, score in neurons:
        d[layer_i][neuron_id].append(pos)
    return {l: dict(nd) for l, nd in d.items()}


def _attn_sets_by_layer(neurons):
    """[(layer_i, head_i, neuron_i, pos, score), ...] -> {layer: frozenset((head_i, neuron_i))}
    Position is intentionally ignored so the same circuit component at different
    source positions is treated as one neuron.
    """
    d = defaultdict(set)
    for layer_i, head_i, neuron_i, pos, score in neurons:
        d[layer_i].add((head_i, neuron_i))
    return {l: frozenset(s) for l, s in d.items()}


def _attn_pos_by_layer(neurons):
    """[(layer_i, head_i, neuron_i, pos, score), ...] -> {layer: {(head_i, neuron_i): [pos, ...]}}"""
    d = defaultdict(lambda: defaultdict(list))
    for layer_i, head_i, neuron_i, pos, score in neurons:
        d[layer_i][(head_i, neuron_i)].append(pos)
    return {l: dict(nd) for l, nd in d.items()}


def _resolve_pos_label(pos, pos_labels):
    """Map a raw position index to its semantic label string."""
    if pos == -1:
        return "last"
    if 0 <= pos < len(pos_labels):
        return pos_labels[pos]
    return "other"


def _jaccard(s_a, s_b):
    u = s_a | s_b
    return len(s_a & s_b) / len(u) if u else float("nan")


# ── Per-prompt-type accumulator ───────────────────────────────────────────────

class _Accum:
    """
    Accumulates per-example statistics for one (prompt_type, subset) combination.

    The compact outputs avoid storing one row per neuron per example:
      neuron_layer_dist.csv    — mean_count per (order, ntype, layer)
      chain_profile.csv        — mean_score per (order, ntype, layer)
      chain_mean_layer.csv     — score-weighted mean layer per (order, ntype)
      chain_position_dist.csv  — mean_count per (order, ntype, position_type)
                                 position_type ∈ {"e1", "e2", "last", "other"}
    """
    def __init__(self):
        self.layer_rows      = []
        self.head_rows       = []
        self.ffn_query_rows  = []
        self.attn_query_rows = []
        self.n_examples      = 0
        # (order, ntype, layer) accumulators
        self.count_sum   = defaultdict(float)   # sum of #neurons per example
        self.score_sum   = defaultdict(float)   # sum of scores (per individual neuron)
        self.score_count = defaultdict(int)     # number of neurons seen
        # (order, ntype) accumulators for mean-layer calculation
        self.wt_layer_sum = defaultdict(float)  # sum(|score| * layer)
        self.wt_sum       = defaultdict(float)  # sum(|score|)
        self.wt_n         = defaultdict(int)    # total neuron count
        # (order, ntype, position_type) accumulators
        self.pos_count = defaultdict(float)     # sum of #neurons at each position type

    def add_neurons(self, order, ntype, neurons, pos_labels=None):
        """Register one example's neuron rows for aggregation.

        pos_labels: list of str (length = seq_len), mapping token index ->
            position type label ("e1", "e2", "last", "other").
            FFN neurons: pos is at index 2 in (layer, neuron_id, pos, score).
            Attn neurons: pos is at index 3 in (layer, head_i, neuron_i, pos, score).
            pos == -1 means the last token (mapped to "last").
        """
        is_ffn  = ntype.startswith("ffn")
        pos_idx = 2 if is_ffn else 3
        counts = defaultdict(int)
        for row in neurons:
            layer_i = row[0]
            score   = row[-1]
            counts[layer_i] += 1
            self.score_sum[(order, ntype, layer_i)]   += score
            self.score_count[(order, ntype, layer_i)] += 1
            w = abs(score)
            self.wt_layer_sum[(order, ntype)] += w * layer_i
            self.wt_sum[(order, ntype)]       += w
            self.wt_n[(order, ntype)]         += 1
            # Position distribution
            if pos_labels is not None:
                pos = row[pos_idx]
                if pos == -1:
                    plabel = "last"
                elif 0 <= pos < len(pos_labels):
                    plabel = pos_labels[pos]
                else:
                    plabel = "other"
                self.pos_count[(order, ntype, plabel)] += 1
        for layer_i, cnt in counts.items():
            self.count_sum[(order, ntype, layer_i)] += cnt

    def save(self, output_dir, n_examples):
        """Write compact CSV files to output_dir."""
        pd.DataFrame(self.layer_rows).to_csv(output_dir / "sublayer_lpi.csv",     index=False)
        pd.DataFrame(self.head_rows).to_csv(output_dir / "head_lpi.csv",           index=False)
        pd.DataFrame(self.ffn_query_rows).to_csv(output_dir / "ffn_query_layers.csv",  index=False)
        pd.DataFrame(self.attn_query_rows).to_csv(output_dir / "attn_query_layers.csv", index=False)

        # Neuron layer distribution
        dist_rows = [
            {"order": order, "ntype": ntype, "layer": layer,
             "mean_count": cnt / n_examples}
            for (order, ntype, layer), cnt in self.count_sum.items()
        ]
        pd.DataFrame(dist_rows).to_csv(output_dir / "neuron_layer_dist.csv", index=False)

        # Chain profile (mean score per layer)
        profile_rows = [
            {"order": order, "ntype": ntype, "layer": layer,
             "mean_score": self.score_sum[(order, ntype, layer)] / n,
             "n": n}
            for (order, ntype, layer), n in self.score_count.items()
        ]
        pd.DataFrame(profile_rows).to_csv(output_dir / "chain_profile.csv", index=False)

        # Score-weighted mean layer per (order, ntype)
        mean_layer_rows = [
            {"order": order, "ntype": ntype,
             "mean_layer": self.wt_layer_sum[(order, ntype)] / wt,
             "n_neurons": self.wt_n[(order, ntype)]}
            for (order, ntype), wt in self.wt_sum.items() if wt > 0
        ]
        pd.DataFrame(mean_layer_rows).to_csv(output_dir / "chain_mean_layer.csv", index=False)

        # Position distribution per (order, ntype, position_type)
        pos_rows = [
            {"order": order, "ntype": ntype, "position": plabel,
             "mean_count": cnt / n_examples}
            for (order, ntype, plabel), cnt in self.pos_count.items()
        ]
        pd.DataFrame(pos_rows).to_csv(output_dir / "chain_position_dist.csv", index=False)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(path, accum, jaccard_accum, overlap_pos_accum,
                     overlap_pos_layer_accum, overlap_pos_n, next_idx):
    with open(path, "wb") as f:
        pickle.dump({"accum": accum, "jaccard_accum": jaccard_accum,
                     "overlap_pos_accum": overlap_pos_accum,
                     "overlap_pos_layer_accum": overlap_pos_layer_accum,
                     "overlap_pos_n": overlap_pos_n,
                     "next_idx": next_idx}, f)


def _load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, clean_up_tokenization_spaces=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    patch_model.patch(model)
    info = patch_model.model_info(model)

    cfg = model.config
    # layer_num must be known before pre-stacking weights; resolve it first
    if hasattr(cfg, "n_layer"):
        _layer_num_tmp = cfg.n_layer
    elif hasattr(cfg, "num_hidden_layers"):
        _layer_num_tmp = cfg.num_hidden_layers
    else:
        raise ValueError(f"Cannot infer layer count from {type(cfg).__name__}")

    # Pre-stack weight tensors once — avoids repeated lambda calls per example/layer
    with torch.no_grad():
        info["fc2_stack"] = torch.stack([
            info["get_ffn_down_proj"](info["layers"][i]) for i in range(_layer_num_tmp)
        ])  # (layer_num, ffn_dim, hidden_dim)
        info["o_proj_stack"] = torch.stack([
            info["get_attn_o_proj"](info["layers"][i]).contiguous() for i in range(_layer_num_tmp)
        ])  # (layer_num, head_num, head_dim, hidden_dim)
    if hasattr(cfg, "n_layer"):
        layer_num    = cfg.n_layer
        head_num     = cfg.n_head
        head_dim     = cfg.n_embd // cfg.n_head
        num_kv_heads = cfg.n_head
    else:
        layer_num    = cfg.num_hidden_layers
        head_num     = cfg.num_attention_heads
        head_dim     = cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads

    data_dir = Path(args.data_dir)
    prompt_pairs = list(combinations(args.prompts, 2))

    for subset in args.subsets:
        input_path = data_dir / f"composition_{subset}" / "two_hop.csv"
        if not input_path.exists():
            print(f"\n[SKIP] {input_path} not found")
            continue

        df = pd.read_csv(input_path, index_col=0)
        n  = len(df) if args.max_examples is None else min(len(df), args.max_examples)
        print(f"\n=== subset={subset}: {n} examples, prompts={args.prompts} ===")

        # Checkpoint: resume from partial run if available
        ckpt_path = data_dir / "neuron_attribution" / f"{subset}_ckpt.pkl"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        start_idx = 0
        accum = {pt: _Accum() for pt in args.prompts}
        jaccard_accum = defaultdict(list)
        overlap_pos_accum       = defaultdict(float)  # key+(side,pos_label) -> total count
        overlap_pos_layer_accum = defaultdict(float)  # key+(layer,side,pos_label) -> total count
        overlap_pos_n           = defaultdict(int)    # key -> n_examples
        if ckpt_path.exists():
            ckpt = _load_checkpoint(ckpt_path)
            accum             = ckpt["accum"]
            jaccard_accum     = ckpt["jaccard_accum"]
            overlap_pos_accum       = ckpt.get("overlap_pos_accum",       defaultdict(float))
            overlap_pos_layer_accum = ckpt.get("overlap_pos_layer_accum", defaultdict(float))
            overlap_pos_n           = ckpt.get("overlap_pos_n",           defaultdict(int))
            start_idx         = ckpt["next_idx"]
            print(f"  Resuming from checkpoint at example {start_idx}/{n}")

        for idx in range(start_idx, n):
            row    = df.iloc[idx]
            row_id = row.get("id", idx)

            # Process every prompt type for this example
            ex_results = {}
            for pt in args.prompts:
                prompt = _get_prompt(row, pt)
                try:
                    result = process_example(
                        model, info, tokenizer, prompt,
                        layer_num, head_num, head_dim, num_kv_heads,
                        top_k_ffn=args.top_k_ffn,
                        top_k_proj=args.top_k_proj,
                        top_k_query=args.top_k_query,
                        max_order=2 if pt == "two-hop" else 1,
                    )
                    ex_results[pt] = result
                except Exception as e:
                    print(f"  [{idx}] {pt} ERROR: {e}", file=sys.stderr)

            # Build neuron sets and update accumulators for each successful prompt type
            ex_sets      = {}   # pt -> {(order, ntype): {layer: frozenset}}
            ex_pos_dicts = {}   # pt -> {(order, ntype): {layer: {key: [pos, ...]}}}
            ex_pos_labels = {}  # pt -> [pos_label, ...]
            for pt, result in ex_results.items():
                layer_rows_ex, head_rows_ex, ffn_top, attn_top, ffq, aq, chain_ex = result
                a = accum[pt]
                a.n_examples += 1

                # Small per-example rows (kept as-is)
                for layer_i, sublayer, lpi, tokens in layer_rows_ex:
                    a.layer_rows.append({"id": row_id, "layer": layer_i, "sublayer": sublayer,
                                         "lpi": lpi, "top_k_tokens": tokens})
                for layer_i, head_i, score, tokens in head_rows_ex:
                    a.head_rows.append({"id": row_id, "layer": layer_i, "head": head_i,
                                        "lpi": score, "top_k_tokens": tokens})
                for layer_i, sublayer, score in ffq:
                    a.ffn_query_rows.append({"id": row_id, "layer": layer_i,
                                             "sublayer": sublayer, "score": score})
                for layer_i, sublayer, score in aq:
                    a.attn_query_rows.append({"id": row_id, "layer": layer_i,
                                              "sublayer": sublayer, "score": score})

                # Compute position labels for this prompt type
                pt_prompt  = _get_prompt(row, pt)
                pt_tokens  = tokenizer.encode(pt_prompt)
                pos_labels = _get_position_labels(pt_tokens, row, tokenizer, pt)
                ex_pos_labels[pt] = pos_labels

                # Aggregate neuron stats with position tracking
                a.add_neurons(0, "ffn",  ffn_top,  pos_labels=pos_labels)
                a.add_neurons(0, "attn", attn_top, pos_labels=pos_labels)
                for order, ntype, rows in chain_ex:
                    a.add_neurons(order, ntype, rows, pos_labels=pos_labels)

                # Build per-layer neuron sets and position dicts for Jaccard computation
                sets = {
                    (0, "ffn"):  _ffn_sets_by_layer(ffn_top),
                    (0, "attn"): _attn_sets_by_layer(attn_top),
                }
                pos_dicts = {
                    (0, "ffn"):  _ffn_pos_by_layer(ffn_top),
                    (0, "attn"): _attn_pos_by_layer(attn_top),
                }
                for order, ntype, rows in chain_ex:
                    is_ffn = ntype.startswith("ffn")
                    sets[(order, ntype)] = (
                        _ffn_sets_by_layer(rows) if is_ffn else _attn_sets_by_layer(rows)
                    )
                    pos_dicts[(order, ntype)] = (
                        _ffn_pos_by_layer(rows) if is_ffn else _attn_pos_by_layer(rows)
                    )
                ex_sets[pt] = sets
                ex_pos_dicts[pt] = pos_dicts

            # Compute Jaccard + overlap position distribution for all prompt-type pairs
            for pt_a, pt_b in prompt_pairs:
                sets_a = ex_sets.get(pt_a, {})
                sets_b = ex_sets.get(pt_b, {})
                if not sets_a or not sets_b:
                    continue
                pair_key = f"{pt_a}__{pt_b}"
                pd_a_all = ex_pos_dicts.get(pt_a, {})
                pd_b_all = ex_pos_dicts.get(pt_b, {})
                pl_a = ex_pos_labels.get(pt_a, [])
                pl_b = ex_pos_labels.get(pt_b, [])
                for (ord_a, ntype_a), layers_a in sets_a.items():
                    nc_a = "ffn" if ntype_a.startswith("ffn") else "attn"
                    pd_a = pd_a_all.get((ord_a, ntype_a), {})
                    for (ord_b, ntype_b), layers_b in sets_b.items():
                        nc_b = "ffn" if ntype_b.startswith("ffn") else "attn"
                        if nc_a != nc_b:
                            continue
                        pd_b = pd_b_all.get((ord_b, ntype_b), {})
                        base_key = (pair_key, nc_a, ntype_a, ord_a, ntype_b, ord_b)
                        for layer in range(layer_num):
                            s_a = layers_a.get(layer, frozenset())
                            s_b = layers_b.get(layer, frozenset())
                            j = _jaccard(s_a, s_b)
                            if not math.isnan(j):
                                jaccard_accum[
                                    base_key + (layer,)
                                ].append(j)
                            # Position distribution of overlapping neurons
                            intersection = s_a & s_b
                            if not intersection:
                                continue
                            nd_a = pd_a.get(layer, {})
                            nd_b = pd_b.get(layer, {})
                            for key in intersection:
                                for pos in nd_a.get(key, []):
                                    lbl = _resolve_pos_label(pos, pl_a)
                                    overlap_pos_accum[base_key + ("a", lbl)] += 1
                                    overlap_pos_layer_accum[base_key + (layer, "a", lbl)] += 1
                                for pos in nd_b.get(key, []):
                                    lbl = _resolve_pos_label(pos, pl_b)
                                    overlap_pos_accum[base_key + ("b", lbl)] += 1
                                    overlap_pos_layer_accum[base_key + (layer, "b", lbl)] += 1
                        # Count this example (once per order-pair, not per layer)
                        overlap_pos_n[base_key] += 1

            if (idx + 1) % 10 == 0 or idx == n - 1:
                print(f"  {idx + 1}/{n}", end="\r", flush=True)
            if args.checkpoint_every > 0 and (idx + 1) % args.checkpoint_every == 0:
                _save_checkpoint(ckpt_path, accum, jaccard_accum,
                                 overlap_pos_accum, overlap_pos_layer_accum,
                                 overlap_pos_n, idx + 1)
        print(f"  {n}/{n} done      ")

        # Save per-prompt-type compact outputs
        na_subset_dir = data_dir / "neuron_attribution" / subset
        for pt in args.prompts:
            a = accum[pt]
            output_dir = na_subset_dir / pt
            output_dir.mkdir(parents=True, exist_ok=True)
            a.save(output_dir, n)
            print(f"  -> {output_dir}/")

        # Save cross-prompt Jaccard CSV in the neuron_attribution subset directory
        jaccard_rows = [
            {"pair": pair, "neuron_class": nc, "ntype_a": ntype_a, "order_a": ord_a,
             "ntype_b": ntype_b, "order_b": ord_b, "layer": layer,
             "mean_jaccard": sum(vals) / len(vals),
             "n": len(vals)}
            for (pair, nc, ntype_a, ord_a, ntype_b, ord_b, layer), vals in jaccard_accum.items()
        ]
        pd.DataFrame(jaccard_rows).to_csv(na_subset_dir / "jaccard.csv", index=False)
        print(f"  -> {na_subset_dir}/jaccard.csv ({len(jaccard_rows)} rows)")

        # Save overlap position distribution
        overlap_pos_rows = [
            {"pair": pair, "neuron_class": nc, "ntype_a": ntype_a, "order_a": ord_a,
             "ntype_b": ntype_b, "order_b": ord_b, "side": side, "position": plabel,
             "mean_count": cnt / max(overlap_pos_n[(pair, nc, ntype_a, ord_a, ntype_b, ord_b)], 1),
             "n": overlap_pos_n[(pair, nc, ntype_a, ord_a, ntype_b, ord_b)]}
            for (pair, nc, ntype_a, ord_a, ntype_b, ord_b, side, plabel), cnt
            in overlap_pos_accum.items()
        ]
        pd.DataFrame(overlap_pos_rows).to_csv(na_subset_dir / "overlap_position.csv", index=False)
        print(f"  -> {na_subset_dir}/overlap_position.csv ({len(overlap_pos_rows)} rows)")

        # Save overlap position × layer distribution
        overlap_pos_layer_rows = [
            {"pair": pair, "neuron_class": nc, "ntype_a": ntype_a, "order_a": ord_a,
             "ntype_b": ntype_b, "order_b": ord_b, "layer": layer, "side": side,
             "position": plabel,
             "mean_count": cnt / max(overlap_pos_n[(pair, nc, ntype_a, ord_a, ntype_b, ord_b)], 1),
             "n": overlap_pos_n[(pair, nc, ntype_a, ord_a, ntype_b, ord_b)]}
            for (pair, nc, ntype_a, ord_a, ntype_b, ord_b, layer, side, plabel), cnt
            in overlap_pos_layer_accum.items()
        ]
        pd.DataFrame(overlap_pos_layer_rows).to_csv(
            na_subset_dir / "overlap_position_by_layer.csv", index=False)
        print(f"  -> {na_subset_dir}/overlap_position_by_layer.csv "
              f"({len(overlap_pos_layer_rows)} rows)")

        # Clean up checkpoint after successful subset completion
        if ckpt_path.exists():
            ckpt_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--data-dir",   default="datasets/gpt2")
    parser.add_argument("--subsets", nargs="+",
                        default=["correct", "incorrect", "inconsistent"],
                        choices=["correct", "incorrect", "inconsistent"])
    parser.add_argument("--prompts", nargs="+",
                        default=["two-hop", "first-hop", "second-hop"],
                        choices=["two-hop", "first-hop", "second-hop"],
                        help="Prompt types to run (default: all three)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--top-k-ffn", type=int, default=300,
                        help="Top-K FFN/attn neurons to keep per example (global across layers)")
    parser.add_argument("--top-k-proj", type=int, default=1,
                        help="Number of top tokens to decode per neuron/head/sublayer projection")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save a resume checkpoint every N examples (0 to disable)")
    parser.add_argument("--top-k-query", type=int, default=300,
                        help="Top-K neurons to track per neuron in backward chain decomposition")
    main(parser.parse_args())
