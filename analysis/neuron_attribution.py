"""
Compute neuron-level LPI and chained backward attribution for all examples.

Replaces neuron_lpi.py + neuron_query.py: loads the model once, loops over all
prompt types and subsets, runs both analyses per example sharing one forward pass.

Outputs per (prompt_type, subset) in the corresponding neuron_attribution_* directory:
  sublayer_lpi.csv        — per-(example, layer, sublayer): lpi + top_k_tokens
  head_lpi.csv            — per-(example, layer, head): lpi + top_k_tokens
  ffn_neurons.csv         — top-K FFN value neurons (order 0): lpi + top_k_tokens
  attn_neurons.csv        — top-K attn value neurons (order 0): lpi + top_k_tokens
  ffn_query_layers.csv    — LPI-weighted residual contribution of prior layers to top-K FFN neurons
  attn_query_layers.csv   — LPI-weighted residual contribution of prior layers to top-K attn neurons
  chain_order{N}_ffn.csv  — order-N FFN neurons in the alternating backward chain
                            (FFN neurons that activate order-(N-1) attn neurons)
  chain_order{N}_attn.csv — order-N attn value neurons in the alternating backward chain
                            (attn neurons that activate order-(N-1) FFN neurons)

  The chain alternates: order-0 FFN/attn → order-1 (attn_query/ffn_query) → order-2 → ...
  Second-hop computation is expected at orders 0-1; first-hop at orders 2-3.

Usage:
    python analysis/neuron_attribution.py
    python analysis/neuron_attribution.py --model-name gpt2 --subsets correct incorrect
    python analysis/neuron_attribution.py --prompts composition first-hop second-hop
    python analysis/neuron_attribution.py --prompts first-hop second-hop --subsets correct incorrect
"""

import argparse
import sys
from pathlib import Path

import patch_model
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Output directory per prompt type ─────────────────────────────────────────

PROMPT_SUBDIR = {
    "composition": "neuron_attribution",
    "first-hop":   "neuron_attribution_first_hop",
    "second-hop":  "neuron_attribution_second_hop",
}


# ── Projection helpers ────────────────────────────────────────────────────────

def project_to_vocab(vectors, final_std, ln_f_weight, lm_head):
    """
    Approximate final-LayerNorm + lm_head projection.

    Uses a fixed normalization scale (final_std) computed from the last layer's
    last-token output, matching the original neuron-attribution paper's approach.

    vectors: (..., hidden_dim)
    returns: (..., vocab_size) logits
    """
    E = vectors.mean(-1, keepdim=True)
    normed = (vectors - E) / final_std * ln_f_weight
    return lm_head(normed)


def decode_top_k(vectors, final_std, ln_f_weight, lm_head, tokenizer, k=10):
    """
    Project vectors through the unembedding and return the top-k token strings.

    vectors: (..., hidden_dim)  — single vector or batch
    returns: list of top-k token strings per vector  (shape: [..., k] as nested list)
    """
    logits = project_to_vocab(vectors, final_std, ln_f_weight, lm_head)
    top_idx = logits.topk(k, dim=-1).indices
    if top_idx.dim() == 1:
        return [tokenizer.decode([i]) for i in top_idx.tolist()]
    flat = top_idx.reshape(-1, k).tolist()
    return [[tokenizer.decode([i]) for i in row] for row in flat]


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
                            final_std, ln_f_weight, lm_head,
                            tokenizer, top_k=300, top_k_proj=10):
    """
    FFN value neuron LPI: per-neuron contribution to the predicted token.

    Returns top_k (layer_i, neuron_id, lpi_score, top_k_tokens) tuples globally,
    sorted descending by score.
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
            all_neurons.append((layer_i, neuron_id, score))

    all_neurons.sort(key=lambda x: x[2], reverse=True)
    all_neurons = all_neurons[:top_k]

    result = []
    for layer_i, neuron_id, score in all_neurons:
        fc2 = info["get_ffn_down_proj"](info["layers"][layer_i])
        tokens = decode_top_k(fc2[neuron_id], final_std, ln_f_weight, lm_head, tokenizer, k=top_k_proj)
        result.append((layer_i, neuron_id, score, tokens))
    return result


def compute_attn_neuron_lpi(states, info, layer_num, head_num, head_dim,
                             predict_index, final_std, ln_f_weight, lm_head,
                             tokenizer, top_k=300, top_k_proj=10):
    """
    Attention value neuron LPI: per-(layer, head, neuron_dim, source_pos) contribution.

    Returns top_k (layer_i, head_i, neuron_id, pos, lpi_score, top_k_tokens) tuples globally,
    sorted descending by score.
    """
    all_neurons = []
    for layer_i in range(layer_num):
        layer_input_last = states[layer_i]["layer_input"][-1]
        v_heads = states[layer_i]["attn_subvalues"].permute(1, 0, 2)
        o_proj  = info["get_attn_o_proj"](info["layers"][layer_i])
        neuron_contribs = v_heads.unsqueeze(-1) * o_proj

        base = torch.log_softmax(project_to_vocab(layer_input_last, final_std, ln_f_weight, lm_head), -1)[predict_index]
        plus   = neuron_contribs + layer_input_last
        logits = project_to_vocab(plus, final_std, ln_f_weight, lm_head)
        lpi    = torch.log_softmax(logits, -1)[..., predict_index] - base

        k = min(top_k, lpi.numel())
        topk_scores, topk_flat_idx = lpi.reshape(-1).topk(k)
        for flat_i, score in zip(topk_flat_idx.tolist(), topk_scores.tolist()):
            pos      = flat_i // (head_num * head_dim)
            head_i   = (flat_i % (head_num * head_dim)) // head_dim
            neuron_i = flat_i % head_dim
            all_neurons.append((layer_i, head_i, neuron_i, pos, score))

    all_neurons.sort(key=lambda x: x[4], reverse=True)
    all_neurons = all_neurons[:top_k]

    result = []
    for layer_i, head_i, neuron_i, pos, score in all_neurons:
        o_proj = info["get_attn_o_proj"](info["layers"][layer_i])
        tokens = decode_top_k(o_proj[head_i, neuron_i], final_std, ln_f_weight, lm_head, tokenizer, k=top_k_proj)
        result.append((layer_i, head_i, neuron_i, pos, score, tokens))
    return result


# ── Query computations ────────────────────────────────────────────────────────

def _build_residual_components(states, layer_limit, pos):
    """
    Collect residual stream components at token position `pos` up to `layer_limit`.

    For FFN query:   layer_limit = ffn_layer, pos = last token (-1)
    For attn query:  layer_limit = attn_layer, pos = source token position
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
    all_scores = [0.0] * (1 + 2 * layer_num)
    for ffn_layer, neuron_id, lpi_score in top_neurons:
        key_scaled = (
            info["get_ffn_key"](info["layers"][ffn_layer], neuron_id)
            * info["get_ffn_ln_weight"](info["layers"][ffn_layer])
        )
        components = _build_residual_components(states, ffn_layer, -1)
        components.append(states[ffn_layer]["attn_output"][-1])
        stacked = torch.stack(components)
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
    all_scores = [0.0] * (1 + 2 * layer_num)
    for attn_layer, head_i, neuron_i, pos, lpi_score in top_neurons:
        kv_head_i   = head_i // num_groups
        flat_neuron = kv_head_i * head_dim + neuron_i
        key_scaled = (
            info["get_attn_v_key"](info["layers"][attn_layer], flat_neuron)
            * info["get_attn_ln_weight"](info["layers"][attn_layer])
        )
        components = _build_residual_components(states, attn_layer, pos)
        stacked = torch.stack(components)
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
                               head_num, num_kv_heads, top_k=300):
    """
    Find which FFN neurons from prior layers most activate the top-K attn value neurons.

    Returns: list of (layer_i, neuron_id, aggregate_score)
    """
    num_groups  = head_num // num_kv_heads
    ffn_scores = {}
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
        for layer_i in range(attn_layer):
            fc2   = info["get_ffn_down_proj"](info["layers"][layer_i])
            coeff = states[layer_i]["coeff_scores"][pos]
            contribs = coeff.unsqueeze(-1) * fc2
            inner  = (contribs * key_scaled).sum(-1)
            normed = inner / total * lpi_score
            k = min(top_k, len(normed))
            for idx in normed.topk(k).indices.tolist():
                key = (layer_i, idx)
                ffn_scores[key] = ffn_scores.get(key, 0.0) + normed[idx].item()

    sorted_neurons = sorted(ffn_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(layer_i, neuron_id, score) for (layer_i, neuron_id), score in sorted_neurons]


def compute_ffn_query_neurons(states, info, top_neurons, head_num, head_dim, top_k=300):
    """
    Find which attention value neurons from prior layers most activate the top-K FFN neurons.

    Symmetric to compute_attn_query_neurons but in the reverse direction:
    - Target: FFN value neurons (ffn_layer, neuron_id, lpi_score)
    - For each target, compute the FFN key projection (gate_proj row scaled by LN weight)
    - Decompose the residual at last token position before FFN into attn value contributions
    - Each attn value neuron (layer_i, head_i, neuron_i, pos) contributes:
          attn_subvalues[head_i, pos, neuron_i] * o_proj[head_i, neuron_i, :]
      to the residual; inner product with key_scaled gives attribution score

    Note: includes ffn_layer's own attention (residual before FFN = layer_input + attn_out).

    Returns: list of (layer_i, head_i, neuron_id, pos, aggregate_score)
    """
    attn_scores = {}
    for ffn_layer, neuron_id, lpi_score in top_neurons:
        key_scaled = (
            info["get_ffn_key"](info["layers"][ffn_layer], neuron_id)
            * info["get_ffn_ln_weight"](info["layers"][ffn_layer])
        )
        # Normalize against the full residual at last token before FFN
        residual = states[ffn_layer]["residual_output"][-1]
        total = (residual * key_scaled).sum().item()
        if abs(total) < 1e-9:
            continue

        # Include attn contributions from layers 0..ffn_layer (inclusive: the FFN sees
        # residual_output = layer_input + attn_out at ffn_layer)
        for layer_i in range(ffn_layer + 1):
            attn_sv = states[layer_i]["attn_subvalues"]  # [heads, seq, head_dim]
            o_proj  = info["get_attn_o_proj"](info["layers"][layer_i])  # [heads, head_dim, hidden]
            # dot_products[h, n] = (o_proj[h, n, :] * key_scaled).sum()
            dot_products = (o_proj * key_scaled).sum(-1)              # [heads, head_dim]
            # weighted[h, pos, n] = attn_sv[h, pos, n] * dot_products[h, n]
            weighted = attn_sv * dot_products.unsqueeze(1)            # [heads, seq, head_dim]
            normed   = weighted / total * lpi_score

            k = min(top_k, normed.numel())
            topk_scores, topk_flat = normed.reshape(-1).topk(k)
            seq_len = attn_sv.shape[1]
            for flat_i, score in zip(topk_flat.tolist(), topk_scores.tolist()):
                head_i   = flat_i // (seq_len * head_dim)
                rem      = flat_i % (seq_len * head_dim)
                pos      = rem // head_dim
                neuron_i = rem % head_dim
                key = (layer_i, head_i, neuron_i, pos)
                attn_scores[key] = attn_scores.get(key, 0.0) + score

    sorted_neurons = sorted(attn_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(layer_i, head_i, neuron_i, pos, score)
            for (layer_i, head_i, neuron_i, pos), score in sorted_neurons]


# ── Per-example driver ────────────────────────────────────────────────────────

def process_example(model, info, tokenizer, prompt, layer_num, head_num, head_dim,
                    num_kv_heads, top_k_ffn, top_k_proj, top_k_query, max_order=None):
    """
    Run one forward pass and compute all LPI and chained backward attribution.

    LPI and query share the same forward pass states — no redundant computation.

    The backward chain alternates FFN→attn and attn→FFN until both produce empty
    neuron sets (chain naturally exhausted) or max_order is reached (safety cap).
    With max_order=None (default) the chain runs to completion.

    Returns:
        layer_rows, head_rows, ffn_rows, attn_neuron_rows,
        ffn_query_rows, attn_query_rows,
        chain   — list of (order, neuron_type, rows) for order 1..depth
                  neuron_type "ffn"  -> [(layer_i, neuron_id, score)]
                  neuron_type "attn" -> [(layer_i, head_i, neuron_id, pos, score)]
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor([tokens], device=device)

    with torch.no_grad():
        model(tokens_tensor, use_cache=False)

    states = patch_model.extract_states(layer_num)

    last_hidden = states[-1]["layer_output"][-1]
    final_std = (last_hidden.var(unbiased=False) + 1e-5).sqrt().item()
    predict_index = info["lm_head"](
        (last_hidden - last_hidden.mean()) / final_std * info["ln_f_weight"]
    ).argmax().item()

    ln_f_weight = info["ln_f_weight"]
    lm_head     = info["lm_head"]

    # LPI
    layer_rows = compute_layer_lpi(
        states, layer_num, predict_index, final_std, ln_f_weight, lm_head,
        tokenizer, top_k_proj=top_k_proj)
    head_rows = compute_head_lpi(
        states, info, layer_num, head_num, head_dim,
        predict_index, final_std, ln_f_weight, lm_head,
        tokenizer, top_k_proj=top_k_proj)
    ffn_rows = compute_ffn_neuron_lpi(
        states, info, layer_num, predict_index,
        final_std, ln_f_weight, lm_head,
        tokenizer, top_k=top_k_ffn, top_k_proj=top_k_proj)
    attn_neuron_rows = compute_attn_neuron_lpi(
        states, info, layer_num, head_num, head_dim,
        predict_index, final_std, ln_f_weight, lm_head,
        tokenizer, top_k=top_k_ffn, top_k_proj=top_k_proj)

    # Layer-level query decomposition — reuse same states
    ffn_top  = [(r[0], r[1], r[2]) for r in ffn_rows]
    attn_top = [(r[0], r[1], r[2], r[3], r[4]) for r in attn_neuron_rows]

    ffn_query_rows  = compute_ffn_query_layers(states, info, ffn_top, layer_num)
    attn_query_rows = compute_attn_query_layers(states, info, attn_top, layer_num, head_dim,
                                                head_num, num_kv_heads)

    # Chained backward attribution: alternates FFN→attn and attn→FFN.
    # Runs until both next sets are empty (chain naturally exhausted at layer 0)
    # or until max_order steps (safety cap; None = no cap).
    chain = []
    cur_ffn  = ffn_top   # [(layer_i, neuron_id, score)]
    cur_attn = attn_top  # [(layer_i, head_i, neuron_i, pos, score)]
    order = 0

    while True:
        order += 1
        if max_order is not None and order > max_order:
            break

        # FFN neurons that activated the previous order's attn neurons
        next_ffn = compute_attn_query_neurons(
            states, info, cur_attn, head_dim, head_num, num_kv_heads, top_k=top_k_query)

        # Attn neurons that activated the previous order's FFN neurons
        next_attn = compute_ffn_query_neurons(
            states, info, cur_ffn, head_num, head_dim, top_k=top_k_query)

        if not next_ffn and not next_attn:
            break  # chain exhausted: nothing left to look back at

        chain.append((order, "ffn", next_ffn))
        chain.append((order, "attn", next_attn))

        cur_ffn  = next_ffn  if next_ffn  else cur_ffn
        cur_attn = next_attn if next_attn else cur_attn

    return (layer_rows, head_rows, ffn_rows, attn_neuron_rows,
            ffn_query_rows, attn_query_rows, chain)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",  # required for Llama (SDPA doesn't expose attn weights)
    ).to(device)
    model.eval()

    patch_model.patch(model)
    info = patch_model.model_info(model)

    cfg = model.config
    if hasattr(cfg, "n_layer"):           # GPT-2
        layer_num    = cfg.n_layer
        head_num     = cfg.n_head
        head_dim     = cfg.n_embd // cfg.n_head
        num_kv_heads = cfg.n_head          # no GQA
    elif hasattr(cfg, "num_hidden_layers"):  # Llama
        layer_num    = cfg.num_hidden_layers
        head_num     = cfg.num_attention_heads
        head_dim     = cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads  # may differ from head_num (GQA)
    else:
        raise ValueError(f"Cannot infer layer/head config from {type(cfg).__name__}")

    data_dir = Path(args.data_dir)

    for prompt_type in args.prompts:
        output_subdir = PROMPT_SUBDIR[prompt_type]
        print(f"\n=== prompt={prompt_type} -> {output_subdir} ===")

        for subset in args.subsets:
            input_path = data_dir / f"composition_{subset}" / "two_hop.csv"
            if not input_path.exists():
                print(f"  [SKIP] {input_path} not found")
                continue

            output_dir = data_dir / f"composition_{subset}" / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(input_path, index_col=0)
            n  = len(df) if args.max_examples is None else min(len(df), args.max_examples)
            print(f"\n  {subset}: {n} examples")

            layer_rows           = []
            head_rows_all        = []
            ffn_rows_all         = []
            attn_neuron_rows_all = []
            ffn_query_all        = []
            attn_query_all       = []
            # chain_data[(order, neuron_type)] -> list of row dicts
            chain_data = {}

            for idx in range(n):
                row = df.iloc[idx]
                if prompt_type == "composition":
                    prompt = row["source_prompt"]
                elif prompt_type == "first-hop":
                    prompt = f"{row['r1_template'].format(row['e1_label'])} is"
                elif prompt_type == "second-hop":
                    prompt = f"{row['r2_template'].format(row['e2_label'])} is"
                row_id = row.get("id", idx)

                try:
                    (layer_rows_ex, head_rows_ex, ffn_rows_ex, attn_neuron_rows_ex,
                     ffq, aq, chain_ex) = process_example(
                        model, info, tokenizer, prompt,
                        layer_num, head_num, head_dim, num_kv_heads,
                        top_k_ffn=args.top_k_ffn,
                        top_k_proj=args.top_k_proj,
                        top_k_query=args.top_k_query,
                        max_order=args.max_order,
                    )
                except Exception as e:
                    print(f"    [{idx}] ERROR: {e}", file=sys.stderr)
                    continue

                for layer_i, sublayer, lpi, tokens in layer_rows_ex:
                    layer_rows.append({"id": row_id, "layer": layer_i, "sublayer": sublayer,
                                       "lpi": lpi, "top_k_tokens": tokens})
                for layer_i, head_i, score, tokens in head_rows_ex:
                    head_rows_all.append({"id": row_id, "layer": layer_i, "head": head_i,
                                          "lpi": score, "top_k_tokens": tokens})
                for layer_i, neuron_id, score, tokens in ffn_rows_ex:
                    ffn_rows_all.append({"id": row_id, "layer": layer_i, "neuron_id": neuron_id,
                                         "lpi": score, "top_k_tokens": tokens})
                for layer_i, head_i, neuron_i, pos, score, tokens in attn_neuron_rows_ex:
                    attn_neuron_rows_all.append({"id": row_id, "layer": layer_i, "head": head_i,
                                                  "neuron_id": neuron_i, "pos": pos,
                                                  "lpi": score, "top_k_tokens": tokens})
                for layer_i, sublayer, score in ffq:
                    ffn_query_all.append({"id": row_id, "layer": layer_i,
                                          "sublayer": sublayer, "score": score})
                for layer_i, sublayer, score in aq:
                    attn_query_all.append({"id": row_id, "layer": layer_i,
                                           "sublayer": sublayer, "score": score})

                for order, ntype, rows in chain_ex:
                    key = (order, ntype)
                    if key not in chain_data:
                        chain_data[key] = []
                    if ntype == "ffn":
                        for layer_i, neuron_id, score in rows:
                            chain_data[key].append({"id": row_id, "layer": layer_i,
                                                    "neuron_id": neuron_id, "score": score})
                    else:  # "attn"
                        for layer_i, head_i, neuron_id, pos, score in rows:
                            chain_data[key].append({"id": row_id, "layer": layer_i,
                                                    "head": head_i, "neuron_id": neuron_id,
                                                    "pos": pos, "score": score})

                if (idx + 1) % 10 == 0 or idx == n - 1:
                    print(f"    {idx + 1}/{n}", end="\r", flush=True)
            print(f"    {n}/{n} done      ")

            pd.DataFrame(layer_rows).to_csv(output_dir / "sublayer_lpi.csv", index=False)
            pd.DataFrame(head_rows_all).to_csv(output_dir / "head_lpi.csv", index=False)
            pd.DataFrame(ffn_rows_all).to_csv(output_dir / "ffn_neurons.csv", index=False)
            pd.DataFrame(attn_neuron_rows_all).to_csv(output_dir / "attn_neurons.csv", index=False)
            pd.DataFrame(ffn_query_all).to_csv(output_dir / "ffn_query_layers.csv", index=False)
            pd.DataFrame(attn_query_all).to_csv(output_dir / "attn_query_layers.csv", index=False)

            for (order, ntype), rows in sorted(chain_data.items()):
                fname = f"chain_order{order}_{ntype}.csv"
                pd.DataFrame(rows).to_csv(output_dir / fname, index=False)

            print(f"    -> {output_dir}/")
            print(f"       sublayer_lpi.csv      : {len(layer_rows)} rows")
            print(f"       head_lpi.csv          : {len(head_rows_all)} rows")
            print(f"       ffn_neurons.csv       : {len(ffn_rows_all)} rows")
            print(f"       attn_neurons.csv      : {len(attn_neuron_rows_all)} rows")
            print(f"       ffn_query_layers.csv  : {len(ffn_query_all)} rows")
            print(f"       attn_query_layers.csv : {len(attn_query_all)} rows")
            for (order, ntype), rows in sorted(chain_data.items()):
                print(f"       chain_order{order}_{ntype}.csv : {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--data-dir",   default="datasets/gpt2")
    parser.add_argument("--subsets", nargs="+",
                        default=["correct", "incorrect", "inconsistent"],
                        choices=["correct", "incorrect", "inconsistent"])
    parser.add_argument("--prompts", nargs="+",
                        default=["composition", "first-hop", "second-hop"],
                        choices=["composition", "first-hop", "second-hop"],
                        help="Prompt types to run (default: all three)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--top-k-ffn", type=int, default=300,
                        help="Top-K FFN/attn neurons to keep per example (global across layers)")
    parser.add_argument("--top-k-proj", type=int, default=5,
                        help="Number of top tokens to decode per neuron/head/sublayer projection")
    parser.add_argument("--top-k-query", type=int, default=300,
                        help="Top-K neurons to track per neuron in backward chain decomposition")
    parser.add_argument("--max-order", type=int, default=None,
                        help="Safety cap on backward attribution chain depth (default: no cap). "
                             "The chain terminates naturally when no new neurons are found. "
                             "Order 1 = direct activators of order-0 neurons; "
                             "second-hop expected at orders 0-1; first-hop at 2-3.")
    main(parser.parse_args())
