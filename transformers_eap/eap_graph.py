# %%

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Callable, Optional
import re
import time
import weakref

# %%

import logging
logging.basicConfig(level=logging.DEBUG)

# %%


class ProfileTracker:
    def __init__(self, device: torch.device, logger=None, stage_width: int = 36):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.stage_width = stage_width
        self.header_printed = False

    def reset_header(self):
        self.header_printed = False

    def begin(self, enabled: bool, reset_peak: bool = True):
        if not enabled:
            return None
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start_alloc = torch.cuda.memory_allocated(self.device)
            start_reserved = torch.cuda.memory_reserved(self.device)
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(self.device)
            return {
                "t0": time.perf_counter(),
                "start_alloc": start_alloc,
                "start_reserved": start_reserved,
            }
        return {
            "t0": time.perf_counter(),
            "start_alloc": None,
            "start_reserved": None,
        }

    def end(self, enabled: bool, start_time, stage_name: str, report_peak: bool = True):
        if not enabled:
            return
        if start_time is None:
            raise ValueError("start_time is None. Ensure that begin() was called and returned a valid timestamp when profiling is enabled.")

        if len(stage_name) > self.stage_width:
            stage_label = stage_name[:self.stage_width - 3] + "..."
        else:
            stage_label = stage_name

        elapsed = time.perf_counter() - start_time["t0"]

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start_alloc = start_time["start_alloc"]
            start_reserved = start_time["start_reserved"]

            end_alloc = torch.cuda.memory_allocated(self.device)
            end_reserved = torch.cuda.memory_reserved(self.device)
            end_alloc_delta = end_alloc - start_alloc
            end_reserved_delta = end_reserved - start_reserved

            if not self.header_printed:
                self.logger.debug(
                    "[PROFILE] "
                    f"{'stage':<{self.stage_width}} {'sec':>7} | "
                    f"{'a_start':>8} {'a_end':>8} {'a_dlt':>8} {'a_peak':>8} | "
                    f"{'r_start':>8} {'r_end':>8} {'r_dlt':>8} {'r_peak':>8}"
                )
                self.header_printed = True

            a_start = start_alloc / (1024 ** 3)
            a_end = end_alloc / (1024 ** 3)
            a_dlt = end_alloc_delta / (1024 ** 3)
            r_start = start_reserved / (1024 ** 3)
            r_end = end_reserved / (1024 ** 3)
            r_dlt = end_reserved_delta / (1024 ** 3)

            if report_peak:
                peak_alloc = torch.cuda.max_memory_allocated(self.device)
                peak_reserved = torch.cuda.max_memory_reserved(self.device)
                peak_alloc_delta = max(0, peak_alloc - start_alloc)
                peak_reserved_delta = max(0, peak_reserved - start_reserved)
                a_peak = peak_alloc_delta / (1024 ** 3)
                r_peak = peak_reserved_delta / (1024 ** 3)

                self.logger.debug(
                    "[PROFILE] "
                    f"{stage_label:<{self.stage_width}} {elapsed:7.3f} | "
                    f"{a_start:8.3f} {a_end:8.3f} {a_dlt:+8.3f} {a_peak:8.3f} | "
                    f"{r_start:8.3f} {r_end:8.3f} {r_dlt:+8.3f} {r_peak:8.3f}"
                )
            else:
                self.logger.debug(
                    "[PROFILE] "
                    f"{stage_label:<{self.stage_width}} {elapsed:7.3f} | "
                    f"{a_start:8.3f} {a_end:8.3f} {a_dlt:+8.3f} {'-':>8} | "
                    f"{r_start:8.3f} {r_end:8.3f} {r_dlt:+8.3f} {'-':>8}"
                )
        else:
            if not self.header_printed:
                self.logger.debug(f"[PROFILE] {'stage':<{self.stage_width}} {'sec':>7}")
                self.header_printed = True
            self.logger.debug(f"[PROFILE] {stage_label:<{self.stage_width}} {elapsed:7.3f}")


# --- Standalone Utility Functions ---

def _parse_node_name(name: str):
    node = {
        "raw": name,
        "layer": None,
        "module": None,
        "head": None,
    }
    match = re.match(r"blocks\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unrecognized node name format: {name}. Expected 'blocks.<layer>.<module>'.")
    node["layer"] = int(match.group(1))
    tail = match.group(2)

    if tail.startswith("attn.hook_result"):
        node["module"] = "attn_result"
        head_match = re.search(r"\[(\d+)\]", tail)
        if not head_match:
            raise ValueError(f"Missing head index in attention result node name: {name}. Expected format 'blocks.<layer>.attn.hook_result[<head>]'.")
        node["head"] = int(head_match.group(1))
    elif "hook_q_input" in tail:
        node["module"] = "q_input"
    elif "hook_k_input" in tail:
        node["module"] = "k_input"
    elif "hook_v_input" in tail:
        node["module"] = "v_input"
    elif "hook_mlp_in" in tail:
        node["module"] = "mlp_in"
    elif "hook_mlp_out" in tail:
        node["module"] = "mlp_out"
    elif "hook_resid_pre" in tail:
        node["module"] = "resid_pre"
    elif "hook_resid_post" in tail:
        node["module"] = "resid_post"
    else:
        raise ValueError(f"Unrecognized module type in node name: {name}. Expected one of 'attn.hook_result', 'hook_q_input', 'hook_k_input', 'hook_v_input', 'hook_mlp_in', 'hook_mlp_out', 'hook_resid_pre', 'hook_resid_post'.")

    return node


def _infer_n_layers_from_scores(scores):
    max_layer = -1
    for key in scores.keys():
        match = re.search(r"blocks\.(\d+)\.", key)
        if match:
            max_layer = max(max_layer, int(match.group(1)))
    if max_layer == -1:
        raise ValueError("Could not infer number of layers from scores. No keys matching pattern 'blocks.<layer>.' found.")
    return max_layer + 1

def format_prompt(
    tokenizer,
    prompts: list[str],
    chat_format: bool = False,
):
    """
    Format prompts for model input. If chat_format is True, wraps prompts in a conversational format.
    """
    if chat_format:
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "Only respond with the answer. Do not include any explanations."},
                {"role": "user", "content": prompt},
            ]
            formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return formatted_prompts
    else:
        return prompts

def encode_prompt(
    tokenizer,
    prompts: list[str],
    subjects: Optional[list[str]] = None,
):
    """
    Find token spans (start, end] for subject strings in each sample.
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    if subjects is not None and len(subjects) != len(prompts):
        raise ValueError(f"Length of subjects must match length of prompts. Got {len(subjects)} subjects and {len(prompts)} prompts.")

    spans = []
    if subjects is not None:
        for i in range(len(prompts)):
            start_char = prompts[i].find(subjects[i])
            end_char = start_char + len(subjects[i])
            
            start_tok = encoded.char_to_token(i, start_char)
            end_tok = encoded.char_to_token(i, end_char)
            
            if start_tok is None or end_tok is None:
                raise ValueError(f"Could not find token span for subject '{subjects[i]}' in prompt '{prompts[i]}'. Check that the subject is a substring of the prompt and that the tokenizer can map characters to tokens correctly.")
            
            spans.append((start_tok, end_tok))

    return encoded, spans

def _make_mlp_hub(layer_idx, score, sample_idx=None, position=None):
    destination_meta = {
        "raw": f"blocks.{int(layer_idx)}.hook_mlp_in",
        "layer": int(layer_idx),
        "module": "mlp_in",
        "head": None,
    }
    return {
        "score": float(score),
        "abs_score": float(abs(score)),
        "source": None,
        "destination": destination_meta,
        "sample_idx": sample_idx,
        "position": position,
    }


def find_top_mlp_hubs_aggregated(scores, n=30):
    """
    Global MLP hub ranking over layers.
    Returns edge-style candidates sorted by absolute score.
    """
    n_layers = _infer_n_layers_from_scores(scores)
    global_hubs = []

    for layer_idx in range(n_layers):
        key = f"blocks.{layer_idx}.hook_mlp_in"
        if key not in scores or not isinstance(scores[key], torch.Tensor):
            raise ValueError(f"Missing or invalid MLP input scores for layer {layer_idx}. Expected a tensor.")
        layer_score = float(scores[key].abs().sum().item())
        global_hubs.append(_make_mlp_hub(layer_idx=layer_idx, score=layer_score, sample_idx=None, position=None))

    return sorted(global_hubs, key=lambda x: x["abs_score"], reverse=True)[:n]


def find_top_mlp_hubs_by_sample(scores, n=30):
    """
    Sample-specific MLP hub ranking over layers.
    Requires per-token MLP score tensors [B, S, Src].
    """
    n_layers = _infer_n_layers_from_scores(scores)

    batch_size = None
    for layer_idx in range(n_layers):
        key = f"blocks.{layer_idx}.hook_mlp_in"
        if key in scores and isinstance(scores[key], torch.Tensor) and scores[key].dim() == 3:
            batch_size = scores[key].size(0)
            break

    if batch_size is None:
        raise ValueError("Per-sample MLP hub analysis requires per-token scores [B, S, Src].")

    sample_hubs = {sample_idx: [] for sample_idx in range(batch_size)}
    for layer_idx in range(n_layers):
        key = f"blocks.{layer_idx}.hook_mlp_in"
        if key not in scores or not isinstance(scores[key], torch.Tensor) or scores[key].dim() != 3:
            raise ValueError(f"Missing or invalid MLP input scores for layer {layer_idx}. Expected [B, S, Src].")

        layer_strength = scores[key].abs().sum(dim=(1, 2))
        for sample_idx in range(batch_size):
            sample_hubs[sample_idx].append(
                _make_mlp_hub(
                    layer_idx=layer_idx,
                    score=layer_strength[sample_idx].item(),
                    sample_idx=int(sample_idx),
                    position=None,
                )
            )

    return {
        sample_idx: sorted(hubs, key=lambda x: x["abs_score"], reverse=True)[:n]
        for sample_idx, hubs in sample_hubs.items()
    }


def find_top_mlp_hubs_by_token_sample(scores, token_level_n=10):
    """
    Sample-specific MLP hub ranking over (token position, layer) pairs.
    Requires per-token MLP score tensors [B, S, Src].
    """
    if token_level_n <= 0:
        raise ValueError(f"token_level_n must be > 0, got {token_level_n}.")

    n_layers = _infer_n_layers_from_scores(scores)

    batch_size = None
    seq_len = None
    for layer_idx in range(n_layers):
        key = f"blocks.{layer_idx}.hook_mlp_in"
        if key in scores and isinstance(scores[key], torch.Tensor) and scores[key].dim() == 3:
            batch_size = scores[key].size(0)
            seq_len = scores[key].size(1)
            break

    if batch_size is None:
        raise ValueError("Token-level MLP hub analysis requires per-token scores [B, S, Src].")

    token_layer_strength = torch.zeros(batch_size, seq_len, n_layers)
    for layer_idx in range(n_layers):
        key = f"blocks.{layer_idx}.hook_mlp_in"
        if key not in scores or not isinstance(scores[key], torch.Tensor) or scores[key].dim() != 3:
            raise ValueError(f"Missing or invalid MLP input scores for layer {layer_idx}. Expected [B, S, Src].")
        token_layer_strength[:, :, layer_idx] = scores[key].abs().sum(dim=-1)

    token_hubs_by_sample = {sample_idx: [] for sample_idx in range(batch_size)}
    for sample_idx in range(batch_size):
        sample_hub_strength = token_layer_strength[sample_idx]  # [S, L]
        k = min(token_level_n, sample_hub_strength.numel())
        if k == 0:
            raise ValueError(f"token_level_n is {token_level_n}, but sample_hub_strength has no elements. Check the shape of your score tensors and the value of token_level_n.")

        top_vals, top_idx = sample_hub_strength.reshape(-1).topk(k)
        for flat_idx, val in zip(top_idx.tolist(), top_vals.tolist()):
            layer_idx = int(flat_idx % n_layers)
            pos_idx = int(flat_idx // n_layers)
            token_hubs_by_sample[sample_idx].append(
                _make_mlp_hub(
                    layer_idx=layer_idx,
                    score=float(val),
                    sample_idx=int(sample_idx),
                    position=int(pos_idx),
                )
            )

    return {
        sample_idx: sorted(hubs, key=lambda x: x["abs_score"], reverse=True)[:token_level_n]
        for sample_idx, hubs in token_hubs_by_sample.items()
    }


def _require_source_names(scores):
    source_names = scores.get("source_names", [])
    if not source_names:
        raise ValueError("Missing 'source_names' in scores. This should be a list of node names corresponding to source indices in the score matrices.")
    return source_names


def _make_edge(source_meta, destination_meta, score, sample_idx=None, position=None):
    return {
        "score": float(score),
        "abs_score": float(abs(score)),
        "source": source_meta,
        "destination": destination_meta,
        "sample_idx": sample_idx,
        "position": position,
    }


def _make_hub(destination_meta, score, sample_idx=None, position=None):
    return {
        "score": float(score),
        "abs_score": float(abs(score)),
        "source": None,
        "destination": destination_meta,
        "sample_idx": sample_idx,
        "position": position,
    }


def find_top_hubs_aggregated(scores, n=30):
    """
    Global hub ranking regardless of module type.
    Aggregates per-token tensors over batch and position, and always sums over source dimension.
    """
    hub_candidates = []

    for dest_name, matrix in scores.items():
        if dest_name == "source_names" or not isinstance(matrix, torch.Tensor):
            continue

        dest_meta = _parse_node_name(dest_name)

        if matrix.dim() == 2:  # [Head, Src]
            n_heads, _ = matrix.shape
            for head_idx in range(n_heads):
                hub_score = matrix[head_idx].sum().item()
                hub_candidates.append(
                    _make_hub(destination_meta={**dest_meta, "head": int(head_idx)}, score=hub_score)
                )
        elif matrix.dim() == 1:  # [Src]
            hub_score = matrix.sum().item()
            hub_candidates.append(_make_hub(destination_meta=dest_meta, score=hub_score))
        elif matrix.dim() == 4:  # [B, Pos, Head, Src]
            agg = matrix.sum(dim=(0, 1))
            n_heads, _ = agg.shape
            for head_idx in range(n_heads):
                hub_score = agg[head_idx].sum().item()
                hub_candidates.append(
                    _make_hub(destination_meta={**dest_meta, "head": int(head_idx)}, score=hub_score)
                )
        elif matrix.dim() == 3:  # [B, Pos, Src]
            agg = matrix.sum(dim=(0, 1))
            hub_score = agg.sum().item()
            hub_candidates.append(_make_hub(destination_meta=dest_meta, score=hub_score))

    return sorted(hub_candidates, key=lambda x: x["abs_score"], reverse=True)[:n]


def find_top_hubs_by_sample(scores, n=30):
    """
    Sample-specific hub ranking regardless of module type.
    Requires per-token score tensors.
    """
    hub_candidates_by_sample = {}

    for dest_name, matrix in scores.items():
        if dest_name == "source_names" or not isinstance(matrix, torch.Tensor):
            continue

        dest_meta = _parse_node_name(dest_name)

        if matrix.dim() == 4:  # [B, Pos, Head, Src]
            bsz, _, n_heads, _ = matrix.shape
            for sample_idx in range(bsz):
                sample_agg = matrix[sample_idx].sum(dim=0)  # [Head, Src]
                for head_idx in range(n_heads):
                    hub_score = sample_agg[head_idx].sum().item()
                    hub_candidates_by_sample.setdefault(int(sample_idx), []).append(
                        _make_hub(
                            destination_meta={**dest_meta, "head": int(head_idx)},
                            score=hub_score,
                            sample_idx=int(sample_idx),
                            position=None,
                        )
                    )
        elif matrix.dim() == 3:  # [B, Pos, Src]
            bsz, _, _ = matrix.shape
            for sample_idx in range(bsz):
                sample_agg = matrix[sample_idx].sum(dim=0)  # [Src]
                hub_score = sample_agg.sum().item()
                hub_candidates_by_sample.setdefault(int(sample_idx), []).append(
                    _make_hub(
                        destination_meta=dest_meta,
                        score=hub_score,
                        sample_idx=int(sample_idx),
                        position=None,
                    )
                )

    return {
        sample_idx: sorted(hubs, key=lambda x: x["abs_score"], reverse=True)[:n]
        for sample_idx, hubs in hub_candidates_by_sample.items()
    }


def find_top_hubs_by_token_sample(scores, token_level_n=10):
    """
    Sample-specific token-level hub ranking regardless of module type.
    Requires per-token score tensors.
    """
    if token_level_n <= 0:
        raise ValueError(f"token_level_n must be > 0, got {token_level_n}.")

    hub_candidates_by_sample = {}

    for dest_name, matrix in scores.items():
        if dest_name == "source_names" or not isinstance(matrix, torch.Tensor):
            continue

        dest_meta = _parse_node_name(dest_name)

        if matrix.dim() == 4:  # [B, Pos, Head, Src]
            bsz, _, n_heads, _ = matrix.shape
            token_strength = matrix.sum(dim=-1)  # [B, Pos, Head]
            for sample_idx in range(bsz):
                sample_flat = token_strength[sample_idx].reshape(-1)
                k = min(token_level_n, sample_flat.numel())
                if k == 0:
                    continue
                top_vals, top_idx = sample_flat.abs().topk(k)
                for abs_val, flat_idx in zip(top_vals.tolist(), top_idx.tolist()):
                    pos_idx = int(flat_idx // n_heads)
                    head_idx = int(flat_idx % n_heads)
                    signed_score = token_strength[sample_idx, pos_idx, head_idx].item()
                    hub_candidates_by_sample.setdefault(int(sample_idx), []).append(
                        _make_hub(
                            destination_meta={**dest_meta, "head": head_idx},
                            score=signed_score,
                            sample_idx=int(sample_idx),
                            position=pos_idx,
                        )
                    )
        elif matrix.dim() == 3:  # [B, Pos, Src]
            bsz, _, _ = matrix.shape
            token_strength = matrix.sum(dim=-1)  # [B, Pos]
            for sample_idx in range(bsz):
                sample_flat = token_strength[sample_idx]
                k = min(token_level_n, sample_flat.numel())
                if k == 0:
                    continue
                top_vals, top_idx = sample_flat.abs().topk(k)
                for abs_val, pos_idx in zip(top_vals.tolist(), top_idx.tolist()):
                    signed_score = token_strength[sample_idx, pos_idx].item()
                    hub_candidates_by_sample.setdefault(int(sample_idx), []).append(
                        _make_hub(
                            destination_meta=dest_meta,
                            score=signed_score,
                            sample_idx=int(sample_idx),
                            position=int(pos_idx),
                        )
                    )

    return {
        sample_idx: sorted(hubs, key=lambda x: x["abs_score"], reverse=True)[:token_level_n]
        for sample_idx, hubs in hub_candidates_by_sample.items()
    }


def find_top_edges_aggregated(scores, n=30):
    """
    Global edge ranking. Uses aggregated tensors directly, and aggregates per-token tensors over batch+position.
    """
    source_names = _require_source_names(scores)
    global_edges = []

    for dest_name, matrix in scores.items():
        if dest_name == "source_names" or not isinstance(matrix, torch.Tensor):
            continue

        dest_meta = _parse_node_name(dest_name)

        if matrix.dim() == 2:  # [Head, Src]
            n_heads, n_srcs = matrix.shape
            for head_idx in range(n_heads):
                for src_idx in range(n_srcs):
                    source_meta = _parse_node_name(source_names[src_idx])
                    global_edges.append(
                        _make_edge(source_meta=source_meta, destination_meta={**dest_meta, "head": int(head_idx)}, score=matrix[head_idx, src_idx].item())
                    )
        elif matrix.dim() == 1:  # [Src]
            n_srcs = matrix.shape[0]
            for src_idx in range(n_srcs):
                source_meta = _parse_node_name(source_names[src_idx])
                global_edges.append(_make_edge(source_meta=source_meta, destination_meta=dest_meta, score=matrix[src_idx].item()))
        elif matrix.dim() == 4:  # [B, Pos, Head, Src]
            agg = matrix.sum(dim=(0, 1))
            n_heads, n_srcs = agg.shape
            for head_idx in range(n_heads):
                for src_idx in range(n_srcs):
                    source_meta = _parse_node_name(source_names[src_idx])
                    global_edges.append(
                        _make_edge(source_meta=source_meta, destination_meta={**dest_meta, "head": int(head_idx)}, score=agg[head_idx, src_idx].item())
                    )
        elif matrix.dim() == 3:  # [B, Pos, Src]
            agg = matrix.sum(dim=(0, 1))
            n_srcs = agg.shape[0]
            for src_idx in range(n_srcs):
                source_meta = _parse_node_name(source_names[src_idx])
                global_edges.append(_make_edge(source_meta=source_meta, destination_meta=dest_meta, score=agg[src_idx].item()))

    top_edges = sorted(global_edges, key=lambda x: x["abs_score"], reverse=True)[:n]
    return top_edges


def find_top_edges_by_sample(scores, n=30):
    """
    Sample-specific edge ranking aggregated over positions only.
    Requires per-token score tensors.
    """
    source_names = _require_source_names(scores)
    sample_edges_by_sample = {}

    for dest_name, matrix in scores.items():
        if dest_name == "source_names" or not isinstance(matrix, torch.Tensor):
            continue
        dest_meta = _parse_node_name(dest_name)

        if matrix.dim() == 4:  # [B, Pos, Head, Src]
            bsz, _, n_heads, n_srcs = matrix.shape
            for sample_idx in range(bsz):
                sample_agg = matrix[sample_idx].sum(dim=0)  # [Head, Src]
                for head_idx in range(n_heads):
                    for src_idx in range(n_srcs):
                        source_meta = _parse_node_name(source_names[src_idx])
                        sample_edges_by_sample.setdefault(int(sample_idx), []).append(
                            _make_edge(
                                source_meta=source_meta,
                                destination_meta={**dest_meta, "head": int(head_idx)},
                                score=sample_agg[head_idx, src_idx].item(),
                                sample_idx=int(sample_idx),
                                position=None,
                            )
                        )
        elif matrix.dim() == 3:  # [B, Pos, Src]
            bsz, _, n_srcs = matrix.shape
            for sample_idx in range(bsz):
                sample_agg = matrix[sample_idx].sum(dim=0)  # [Src]
                for src_idx in range(n_srcs):
                    source_meta = _parse_node_name(source_names[src_idx])
                    sample_edges_by_sample.setdefault(int(sample_idx), []).append(
                        _make_edge(
                            source_meta=source_meta,
                            destination_meta=dest_meta,
                            score=sample_agg[src_idx].item(),
                            sample_idx=int(sample_idx),
                            position=None,
                        )
                    )

    return {
        sample_idx: sorted(edges, key=lambda x: x["abs_score"], reverse=True)[:n]
        for sample_idx, edges in sample_edges_by_sample.items()
    }


def find_top_edges_by_token_sample(scores, token_level_n=10):
    """
    Sample-specific token-level edge ranking.
    Requires per-token score tensors.
    """
    source_names = _require_source_names(scores)
    token_edges_by_sample = {}

    for dest_name, matrix in scores.items():
        if dest_name == "source_names" or not isinstance(matrix, torch.Tensor):
            continue
        dest_meta = _parse_node_name(dest_name)

        if matrix.dim() == 4:  # [B, Pos, Head, Src]
            bsz, _, n_heads, n_srcs = matrix.shape
            for sample_idx in range(bsz):
                sample_abs = matrix[sample_idx].abs()  # [Pos, Head, Src]
                k_tok = min(token_level_n, sample_abs.numel())
                if k_tok == 0:
                    raise ValueError(f"token_level_n is {token_level_n}, but sample_abs has no elements. Check the shape of your score tensors and the value of token_level_n.")
                top_vals, top_idx = sample_abs.reshape(-1).topk(k_tok)
                for abs_val, flat_idx in zip(top_vals.tolist(), top_idx.tolist()):
                    src_idx = flat_idx % n_srcs
                    tmp = flat_idx // n_srcs
                    head_idx = tmp % n_heads
                    pos_idx = tmp // n_heads
                    source_meta = _parse_node_name(source_names[src_idx])
                    token_edges_by_sample.setdefault(int(sample_idx), []).append(
                        _make_edge(
                            source_meta=source_meta,
                            destination_meta={**dest_meta, "head": int(head_idx)},
                            score=matrix[sample_idx, pos_idx, head_idx, src_idx].item(),
                            sample_idx=int(sample_idx),
                            position=int(pos_idx),
                        )
                    )
        elif matrix.dim() == 3:  # [B, Pos, Src]
            bsz, _, n_srcs = matrix.shape
            for sample_idx in range(bsz):
                sample_abs = matrix[sample_idx].abs()  # [Pos, Src]
                k_tok = min(token_level_n, sample_abs.numel())
                if k_tok == 0:
                    raise ValueError(f"token_level_n is {token_level_n}, but sample_abs has no elements. Check the shape of your score tensors and the value of token_level_n.")
                top_vals, top_idx = sample_abs.reshape(-1).topk(k_tok)
                for abs_val, flat_idx in zip(top_vals.tolist(), top_idx.tolist()):
                    src_idx = flat_idx % n_srcs
                    pos_idx = flat_idx // n_srcs
                    source_meta = _parse_node_name(source_names[src_idx])
                    token_edges_by_sample.setdefault(int(sample_idx), []).append(
                        _make_edge(
                            source_meta=source_meta,
                            destination_meta=dest_meta,
                            score=matrix[sample_idx, pos_idx, src_idx].item(),
                            sample_idx=int(sample_idx),
                            position=int(pos_idx),
                        )
                    )

    return {
        sample_idx: sorted(edges, key=lambda x: x["abs_score"], reverse=True)[:token_level_n]
        for sample_idx, edges in token_edges_by_sample.items()
    }

# --- Model Component Registries ---

def get_gpt2_components(model, layer_idx):
    layer = model.transformer.h[layer_idx]
    embed = model.transformer.wte
    return {
        "layer_block": layer,
        "ln_1": layer.ln_1, # Component responsible for Norm before Attention
        "qkv": layer.attn.c_attn,
        "o": layer.attn.c_proj,
        "ln_2": layer.ln_2, # Component responsible for Norm before MLP
        "mlp_in": layer.mlp.c_fc,
        "mlp_out": layer.mlp.c_proj,
    }, embed

def get_llama_like_components(model, layer_idx):
    layer = model.model.layers[layer_idx]
    embed = model.model.embed_tokens
    return {
        "layer_block": layer,
        "ln_1": layer.input_layernorm, # Component responsible for Norm before Attention
        "q": layer.self_attn.q_proj,
        "k": layer.self_attn.k_proj,
        "v": layer.self_attn.v_proj,
        "o": layer.self_attn.o_proj,
        "ln_2": layer.post_attention_layernorm, # MLP Norm
        "mlp_in": layer.mlp, # Use full MLP block to capture gradients from both Gate and Up paths
        "mlp_out": layer.mlp.down_proj,
    }, embed

COMPONENT_REGISTRY = {
    "GPT2LMHeadModel": get_gpt2_components,
    "LlamaForCausalLM": get_llama_like_components,
    "Qwen3ForCausalLM": get_llama_like_components,
}

def get_gpt2_config(model):
    config = model.config
    c = {
        "is_qkv_fused": True,
        "is_qkv_conv1d": True,
        "n_layers": config.n_layer,
        "n_heads": config.n_head,
        "n_kv_heads": config.n_head,
        "hidden_size": config.n_embd,
        "head_dim": config.n_embd // config.n_head,
    }
    n_src = 1 + (c["n_layers"] * c["n_heads"]) + c["n_layers"]
    c["n_src"] = n_src
    return c

def get_llama_like_config(model):
    config = model.config
    c = {
        "is_qkv_fused": False,
        "is_qkv_conv1d": False,
        "n_layers": config.num_hidden_layers,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": config.num_key_value_heads,
        "hidden_size": config.hidden_size,
        "head_dim": config.head_dim,
    }
    n_src = 1 + (c["n_layers"] * c["n_heads"]) + c["n_layers"]
    c["n_src"] = n_src
    return c

CONFIG_REGISTRY = {
    "GPT2LMHeadModel": get_gpt2_config,
    "LlamaForCausalLM": get_llama_like_config,
    "Qwen3ForCausalLM": get_llama_like_config,
}

# --- Metric Factories ---

def get_nll_metric(ans_ids, batch_indices=None):
    """
    Returns a metric function that computes the Negative Log Likelihood (NLL) of the target tokens.
    ans_ids: [Batch] Tensor of target token IDs.
    batch_indices: [Batch] Tensor of batch indices (optional, inferred if None).
    """
    def nll_metric(logits, corrupted_logits=None, input_length: Optional[torch.Tensor] = None):
        # logits: [Batch, Seq, Vocab]
        nonlocal batch_indices
        if batch_indices is None:
            batch_indices = torch.arange(logits.shape[0], device=logits.device)

        selected_logits = _gather_last_token_logits(logits, input_length)
        return -selected_logits.log_softmax(dim=-1)[batch_indices, ans_ids].mean()
    return nll_metric

def _gather_last_token_logits(logits: torch.Tensor, input_length: Optional[torch.Tensor]):
    if input_length is None:
        return logits[:, -1, :]
    input_length = input_length.to(logits.device)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_indices, input_length - 1, :]

def get_logit_diff_metric(ans_ids, foil_ids, batch_indices=None):
    """
    Returns a metric function that computes the Logit Difference (Target - Foil).
    ans_ids: [Batch] Tensor of target token IDs.
    foil_ids: [Batch] Tensor of foil token IDs.
    batch_indices: [Batch] Tensor of batch indices (optional, inferred if None).
    """
    def logit_diff_metric(logits, corrupted_logits=None, input_length: Optional[torch.Tensor] = None):
        # logits: [Batch, Seq, Vocab]
        nonlocal batch_indices
        if batch_indices is None:
            batch_indices = torch.arange(logits.shape[0], device=logits.device)

        selected_logits = _gather_last_token_logits(logits, input_length)
        target_logits = selected_logits[batch_indices, ans_ids]
        foil_logits = selected_logits[batch_indices, foil_ids]
        return (target_logits - foil_logits).mean()
    return logit_diff_metric

def get_kl_div_metric():
    """
    Returns a metric function that computes KL(current || corrupted)
    on the final-token distribution.
    """
    def kl_div_metric(logits, target_logits, input_length: Optional[torch.Tensor] = None):
        log_probs = _gather_last_token_logits(logits, input_length).log_softmax(dim=-1)
        target_log_probs = _gather_last_token_logits(target_logits, input_length).log_softmax(dim=-1).detach()
        return F.kl_div(
            log_probs,
            target_log_probs.to(log_probs.device),
            log_target=True,
            reduction="batchmean",
        )

    return kl_div_metric

class EAPGraph:
    def __init__(self, model):
        """
        model: Pre-loaded HuggingFace model
        graph_type: Optional string ('gpt2', 'llama_like', 'gptj'). If None, mostly auto-detected.
        """
        self.model = model
        
        if hasattr(self.model, "hf_device_map"):
            # For sharded models, main_device is a bit murky, but we need *some* device for tensors
            d = list(self.model.hf_device_map.values())[0]
            # Handle string outputs like 'cpu' or 'cuda:0'
            if isinstance(d, str) or isinstance(d, int):
                self.main_device = torch.device(d)
            else:
                self.main_device = d
        else:
            self.main_device = self.model.device
        
        # Arch detection
        self.arch_type = self._detect_architecture()

        self.config = self._get_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized EAPGraph for {self.arch_type}. Layers: {self.config['n_layers']}, Heads: {self.config['n_heads']}")
        
        # Storage
        self.activations = {}
        self.gradients = {}
        self.norm_io = {}
        self.handles = []
        
        self.profiler = ProfileTracker(self.main_device)

    def _detect_architecture(self):
        name = self.model.__class__.__name__
        if name in COMPONENT_REGISTRY:
            return name
        raise ValueError(f"Could not detect architecture from model class name '{name}'. Please specify graph_type explicitly.")

    def _get_config(self):
        name = self.model.__class__.__name__
        if name not in CONFIG_REGISTRY:
            raise ValueError(f"No config getter registered for architecture '{name}'. Available: {list(CONFIG_REGISTRY.keys())}")
        getter = CONFIG_REGISTRY[self.arch_type]
        return getter(self.model)

    def _get_layer_components(self, layer_idx):
        getter = COMPONENT_REGISTRY[self.arch_type]
        return getter(self.model, layer_idx)

    def reset_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self.norm_io = {}

    def _norm_vjp(self, norm_key: str, grad_out: torch.Tensor):
        if norm_key not in self.norm_io:
            raise KeyError(f"Missing normalization cache for {norm_key}")

        norm_input = self.norm_io[norm_key]["input"]
        norm_output = self.norm_io[norm_key]["output"]

        # shape [B, S, D]
        if grad_out.dim() == 3:
            return torch.autograd.grad(
                outputs=norm_output,
                inputs=norm_input,
                grad_outputs=grad_out,
                retain_graph=True,
                allow_unused=False,
            )[0]

        # shape [B, S, H, D]
        elif grad_out.dim() == 4:
            per_head = []
            for head_idx in range(grad_out.size(2)):
                grad_head = torch.autograd.grad(
                    outputs=norm_output,
                    inputs=norm_input,
                    grad_outputs=grad_out[:, :, head_idx, :],
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                per_head.append(grad_head.unsqueeze(2))
            return torch.cat(per_head, dim=2)

        raise ValueError(f"Unsupported grad_out rank {grad_out.dim()} for norm VJP")

    # --- Activation Hooks ---
    def get_activation_hook(self, name):
        """
        name: string in format "blocks.{layer_idx}.{component}.hook_type"
        
        Examples:
        - "blocks.3.hook_resid_pre"
        - "blocks.5.attn.hook_result"
        - "blocks.2.hook_mlp_out"
        - "blocks.4.ln_1" (captures both input and output for pre-attention norm)
        - "blocks.4.ln_2" (captures both input and output for pre-MLP norm)
        """
        layer_idx = int(name.split(".")[1])
        hook_type = name.split(".")[-1]

        def hook(module, input, output=None): 
            # Pre-hooks: (module, input) -> None/Modified Input
            # Post-hooks: (module, input, output) -> None/Modified Output
            
            if hook_type in ["hook_resid_pre"]:
                self.activations[name] = input[0].detach()
            elif hook_type == "hook_result":
                x = input[0].detach()
                b, s, _ = x.shape
                x = x.view(b, s, self.config["n_heads"], self.config["head_dim"])
                
                comps, _ = self._get_layer_components(layer_idx)
                w_o = comps["o"].weight
                w_o = w_o.view(self.config["hidden_size"], self.config["n_heads"], self.config["head_dim"])
                
                self.activations[name] = torch.einsum("bshd,ohd->bsho", x, w_o).detach()
            elif hook_type in ["hook_mlp_out"]:
                self.activations[name] = output.detach()
            
            # Necessary for accurate Attention Head and MLP attribution
            elif hook_type in ["ln_1", "ln_2"]:
                self.norm_io[name] = {
                    "input": input[0],
                    "output": output,
                }
            else:
                raise ValueError(f"Unsupported hook type '{hook_type}' for activation hook '{name}'.")

        return hook

    # --- Gradient Hooks ---
    # def get_gradient_hook(self, name):
    def get_gradient_hook(self, name, type="std", layer_idx=None):
        """
        Retrieves gradients for attention heads and MLPs.
        
        For Attention/MLP destinations, this function backpropagates through normalization
        using autograd VJPs from cached norm module input/output tensors, which keeps
        attribution model-agnostic across LayerNorm/RMSNorm variants.
        
        This enables proper attribution of previous layer outputs -> current layer heads.
        """
        hook_type = name.split(".")[-1]
        layer_idx = int(name.split(".")[1])
        
        def hook(module, grad_input, grad_output):
            if hook_type == "hook_attn_in":                
                w = module.weight
                if self.config["is_qkv_conv1d"]:
                    w_q, w_k, w_v = w.t().split(self.config["hidden_size"], dim=0)
                    w_q = w_q.view(self.config["n_heads"], self.config["head_dim"], self.config["hidden_size"])
                    w_k = w_k.view(self.config["n_kv_heads"], self.config["head_dim"], self.config["hidden_size"])
                    w_v = w_v.view(self.config["n_kv_heads"], self.config["head_dim"], self.config["hidden_size"])
                else:
                    raise NotImplementedError("hook_attn_in gradient hook is only implemented for fused Conv1D QKV (e.g. GPT-2).")

                gy = grad_output[0]
                b, s, _ = gy.shape
                gy_q, gy_k, gy_v = gy.split(self.config["hidden_size"], dim=-1)
                gy_q = gy_q.view(b, s, self.config["n_heads"], self.config["head_dim"])
                gy_k = gy_k.view(b, s, self.config["n_kv_heads"], self.config["head_dim"])
                gy_v = gy_v.view(b, s, self.config["n_kv_heads"], self.config["head_dim"])
                
                gx_q = torch.einsum("bsho,hoi->bshi", gy_q, w_q)
                gx_k = torch.einsum("bsho,hoi->bshi", gy_k, w_k)
                gx_v = torch.einsum("bsho,hoi->bshi", gy_v, w_v)
                
                gx_q = self._norm_vjp(f"blocks.{layer_idx}.ln_1", gx_q)
                gx_k = self._norm_vjp(f"blocks.{layer_idx}.ln_1", gx_k)
                gx_v = self._norm_vjp(f"blocks.{layer_idx}.ln_1", gx_v)
                
                self.gradients[f"blocks.{layer_idx}.hook_q_input"] = gx_q.detach().cpu()
                self.gradients[f"blocks.{layer_idx}.hook_k_input"] = gx_k.detach().cpu()
                self.gradients[f"blocks.{layer_idx}.hook_v_input"] = gx_v.detach().cpu()
                
            elif hook_type in ["hook_q_input", "hook_k_input", "hook_v_input"]:
                if hook_type == "hook_q_input":
                    n_heads = self.config["n_heads"]
                else:
                    n_heads = self.config["n_kv_heads"]
                
                gy = grad_output[0]
                b, s, _ = gy.shape
                gy = gy.view(b, s, n_heads, self.config["head_dim"])
                
                w = module.weight
                w = w.view(n_heads, self.config["head_dim"], self.config["hidden_size"])
                
                gx = torch.einsum("bsho,hoi->bshi", gy, w)
                gx = self._norm_vjp(f"blocks.{layer_idx}.ln_1", gx)
                self.gradients[name] = gx.detach().cpu()
                
            elif hook_type == "hook_mlp_in":
                gx = grad_input[0]
                gx = self._norm_vjp(f"blocks.{layer_idx}.ln_2", gx)
                self.gradients[name] = gx.detach().cpu()
            
            elif hook_type == "hook_resid_post":
                self.gradients[name] = grad_output[0].detach().cpu()
            
            else:
                raise ValueError(f"Unsupported hook type '{hook_type}' for gradient hook '{name}'.")

        return hook

    def register_forward_hooks(self):
        self.reset_hooks()
        for i in range(self.config["n_layers"]):
            comps, _ = self._get_layer_components(i)
            
            # 1. Resid Pre (Block Input)
            self.handles.append(comps["layer_block"].register_forward_pre_hook(
                 self.get_activation_hook(f"blocks.{i}.hook_resid_pre")
            ))
            
            # 2. Head Results (Source) - Hook Input to O_proj
            self.handles.append(comps["o"].register_forward_pre_hook(
                self.get_activation_hook(f"blocks.{i}.attn.hook_result")
            ))

            # 3. MLP Out (Source)
            self.handles.append(comps["mlp_out"].register_forward_hook(
                self.get_activation_hook(f"blocks.{i}.hook_mlp_out")
            ))
            
            # 4. Attention Pre-LN Input (Resid Pre)
            # Necessary for accurate Attention Head attribution
            self.handles.append(comps["ln_1"].register_forward_hook(
                self.get_activation_hook(f"blocks.{i}.ln_1")
            ))
            
            # 5. MLP Pre-LN Input (Resid Mid)
            # Necessary for accurate MLP attribution
            if "ln_2" in comps and comps["ln_2"] is not None:
                self.handles.append(comps["ln_2"].register_forward_hook(
                    self.get_activation_hook(f"blocks.{i}.ln_2")
                ))

    def register_backward_hooks(self):
        for i in range(self.config["n_layers"]):
            comps, _ = self._get_layer_components(i)
            
            # 1. Q, K, V Inputs
            if self.config["is_qkv_fused"]:
                self.handles.append(comps["qkv"].register_full_backward_hook(
                    self.get_gradient_hook(f"blocks.{i}.hook_attn_in")
                ))
            else:
                self.handles.append(comps["q"].register_full_backward_hook(
                    self.get_gradient_hook(f"blocks.{i}.hook_q_input")
                ))
                self.handles.append(comps["k"].register_full_backward_hook(
                    self.get_gradient_hook(f"blocks.{i}.hook_k_input")
                ))
                self.handles.append(comps["v"].register_full_backward_hook(
                    self.get_gradient_hook(f"blocks.{i}.hook_v_input")
                ))
            
            # 2. MLP Input
            self.handles.append(comps["mlp_in"].register_full_backward_hook(
                self.get_gradient_hook(f"blocks.{i}.hook_mlp_in")
            ))
            
            # 3. Resid Post
            self.handles.append(comps["layer_block"].register_full_backward_hook(
                self.get_gradient_hook(f"blocks.{i}.hook_resid_post")
            ))

    def _prepare_activations(
        self,
        input_ids,
        attention_mask,
        corrupted_input_ids,
        corrupted_attention_mask,
        subject_spans,
        profile = False,
    ):
        phase_start = self.profiler.begin(profile, reset_peak=True)
    
        use_subject_noise_baseline = corrupted_input_ids is None

        if use_subject_noise_baseline and subject_spans is None:
            raise ValueError("subject_spans is required when corrupted_input_ids is None for the noise baseline approach.")
            
        if not use_subject_noise_baseline and (corrupted_input_ids is None or corrupted_attention_mask is None):
            raise ValueError("corrupted_input_ids and corrupted_attention_mask are required when not using the noise baseline approach.")

        if corrupted_attention_mask is not None:
            # Provide a clearer error message listing which samples differ in length
            input_lengths = attention_mask.sum(-1).cpu()
            corrupted_lengths = corrupted_attention_mask.sum(-1).cpu()
            unequal = input_lengths != corrupted_lengths
            if torch.any(unequal):
                mismatched = torch.where(unequal)[0].tolist()
                raise ValueError(
                    f"Pair of clean and corrupted prompts must have the same token length. "
                    f"Mismatched sample indices: {mismatched}. "
                    f"Clean lengths: {input_lengths.tolist()}, Corrupted lengths: {corrupted_lengths.tolist()}"
                )

        self.model.eval()
        self.reset_hooks()
        
        input_ids = input_ids.to(self.main_device)
        attention_mask = attention_mask.to(self.main_device)
        input_length = attention_mask.sum(dim=-1)
        if not use_subject_noise_baseline:
            corrupted_input_ids = corrupted_input_ids.to(self.main_device)
        
        self.register_forward_hooks()
        with torch.no_grad():
            self.model(input_ids)
            clean_acts = self.activations.copy()
        self.reset_hooks()

        _, embed = self._get_layer_components(0)
        embed_acts = []

        def stat_hook(m, i, o):
            embed_acts.append(o.detach())

        h = embed.register_forward_hook(stat_hook)
        with torch.no_grad():
            clean_logits = self.model(input_ids).logits.detach()
        h.remove()
        clean_wte = embed_acts[0]

        if not use_subject_noise_baseline:
            embed_acts = []
            h = embed.register_forward_hook(stat_hook)
            self.register_forward_hooks()
            with torch.no_grad():
                corrupted_logits = self.model(corrupted_input_ids).logits.detach()
                corrupt_acts = self.activations.copy()
            h.remove()
            self.reset_hooks()

            corrupted_wte = embed_acts[0]
        else:
            subject_tokens = []
            for i, (start, end) in enumerate(subject_spans):
                subject_tokens.append(clean_wte[i, start:end]) # shape [Subject_Tokens, D_model]
            subject_tokens = torch.cat(subject_tokens, dim=0) # [Total_Subject_Tokens, D_model]
            std = subject_tokens.std().item()
            corrupted_wte = clean_wte.clone()
            for i, (start, end) in enumerate(subject_spans):
                corrupted_wte[i, start:end] += std

            def noisy_embed_hook(module, input, output):
                return corrupted_wte

            h = embed.register_forward_hook(noisy_embed_hook)
            self.register_forward_hooks()
            with torch.no_grad():
                corrupted_logits = self.model(input_ids).logits.detach()
                corrupt_acts = self.activations.copy()
            h.remove()
            self.reset_hooks()

        activation_differences = {}
        for name, clean_act in clean_acts.items():
            if name not in corrupt_acts:
                raise ValueError(f"Missing activation for '{name}' in corrupted forward pass.")
            if clean_act.dim() == 3:
                mask = attention_mask.unsqueeze(-1).to(clean_act.device)
            elif clean_act.dim() == 4:
                mask = attention_mask.unsqueeze(-1).unsqueeze(-1).to(clean_act.device)
            else:
                raise ValueError(f"Unsupported activation rank {clean_act.dim()} for '{name}'.")
            activation_differences[name] = (corrupt_acts[name] - clean_act) * mask

        self.profiler.end(profile, phase_start, "prepare_activations", report_peak=True)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_length": input_length,
            "clean_wte": clean_wte,
            "corrupted_wte": corrupted_wte,
            "clean_logits": clean_logits,
            "corrupted_logits": corrupted_logits,
            "activation_differences": activation_differences,
        }

    def _compute_gradients(self, input_ids, input_length, clean_wte, corrupted_wte, metric_fn: Callable, reference_logits, steps: int, profile = False):
        phase_start = self.profiler.begin(profile, reset_peak=True)
        _, embed = self._get_layer_components(0)
        avg_grads = {}

        for step in range(1, steps + 1):
            step_tag = f"compute_gradients.step{step}/{steps}"

            self.reset_hooks()
            self.model.zero_grad()
            self.register_forward_hooks()
            self.register_backward_hooks()

            def interpolation_hook(module, input, output):
                interpolated = corrupted_wte + (step / steps) * (clean_wte - corrupted_wte)
                interpolated.requires_grad_(True)
                return interpolated

            h_interp = embed.register_forward_hook(interpolation_hook)

            fwd_start = self.profiler.begin(profile, reset_peak=False)
            logits = self.model(input_ids).logits
            loss = metric_fn(logits, reference_logits, input_length)
            self.profiler.end(profile, fwd_start, f"{step_tag}.forward_loss", report_peak=False)

            bwd_start = self.profiler.begin(profile, reset_peak=False)
            loss.backward()
            self.profiler.end(profile, bwd_start, f"{step_tag}.backward", report_peak=False)
            h_interp.remove()

            acc_start = self.profiler.begin(profile, reset_peak=False)
            for name, grad in self.gradients.items():
                if name not in avg_grads:
                    avg_grads[name] = grad.detach().cpu()
                else:
                    avg_grads[name] += grad
            self.profiler.end(profile, acc_start, f"{step_tag}.accumulate", report_peak=False)

        for name in avg_grads:
            avg_grads[name] /= steps

        self.profiler.end(profile, phase_start, "compute_gradients", report_peak=True)
        return avg_grads

    def get_activations(
        self,
        input_ids,
        attention_mask,
        corrupted_input_ids,
        corrupted_attention_mask,
        subject_spans,
        metric_fn: Callable,
        profile = False,
    ):
        prepared = self._prepare_activations(
            input_ids,
            attention_mask,
            corrupted_input_ids,
            corrupted_attention_mask,
            subject_spans,
            profile=profile,
        )
        gradients = self._compute_gradients(
            prepared["input_ids"],
            prepared["input_length"],
            prepared["clean_wte"],
            prepared["corrupted_wte"],
            metric_fn,
            prepared["corrupted_logits"],
            steps=1,
            profile=profile,
        )
        return prepared["activation_differences"], gradients

    def get_activations_ig(
        self,
        input_ids,
        attention_mask,
        corrupted_input_ids,
        corrupted_attention_mask,
        subject_spans,
        metric_fn: Callable,
        steps: int = 30,
        profile = False,
    ):
        if steps <= 0:
            raise ValueError(f"integrated_gradients must be > 0, got {steps}")

        prepared = self._prepare_activations(
            input_ids,
            attention_mask,
            corrupted_input_ids,
            corrupted_attention_mask,
            subject_spans,
            profile=profile,
        )
        gradients = self._compute_gradients(
            prepared["input_ids"],
            prepared["input_length"],
            prepared["clean_wte"],
            prepared["corrupted_wte"],
            metric_fn,
            prepared["clean_logits"],
            steps=steps,
            profile=profile,
        )
        return prepared["activation_differences"], gradients

    def attribute(
        self,
        input_ids,
        attention_mask,
        metric_fn: Callable,
        integrated_gradients: Optional[int] = None,
        corrupted_input_ids: Optional[torch.Tensor] = None,
        corrupted_attention_mask: Optional[torch.Tensor] = None,
        subject_spans: Optional[list[tuple[int, int]]] = None,
        return_per_token_scores: bool = False,
        profile: bool = False,
    ):
        """
        Core EAP/EAP-IG execution using pre-processed tensors.
        input_ids: [Batch, Seq]
        attention_mask: [Batch, Seq]
        metric_fn: Callable that maps (logits, reference_logits, input_length) -> scalar metric/loss.
        integrated_gradients: If None, run EAP. If int>0, run EAP-IG with this many steps.
        corrupted_input_ids: [Batch, Seq] corrupted baseline inputs.
            If None, a prompts-only baseline is created by noising subject token embeddings.
        corrupted_attention_mask: [Batch, Seq] attention mask for corrupted_input_ids.
        subject_spans: Optional[list[tuple[int, int]]] = None,
        return_per_token_scores: bool = False,
        profile: if True, print phase runtime and CUDA peak memory.
        """
        if profile:
            self.profiler.reset_header()

        if integrated_gradients is None:
            activation_differences, clean_grads = self.get_activations(
                input_ids,
                attention_mask,
                corrupted_input_ids,
                corrupted_attention_mask,
                subject_spans,
                metric_fn,
                profile=profile,
            )
        else:
            activation_differences, clean_grads = self.get_activations_ig(
                input_ids,
                attention_mask,
                corrupted_input_ids,
                corrupted_attention_mask,
                subject_spans,
                metric_fn,
                steps=integrated_gradients,
                profile=profile,
            )

        phase_start = self.profiler.begin(profile, reset_peak=True)
        
        resid_pre_name = "blocks.0.hook_resid_pre"
        if resid_pre_name not in activation_differences:
            raise ValueError("Missing required activation difference for blocks.0.hook_resid_pre")

        # shape [Batch, Seq, Dest, Src]
        scores = {}
        
        # 1. Source: Resid Pre Layer 0
        # activation_difference[resid_pre_name] shape [Batch, Seq, D_model]
        d0 = activation_differences[resid_pre_name]

        bsz, seq_len, _ = d0.shape
        src_buffer = torch.empty(
            (bsz, seq_len, self.config["n_src"], self.config["hidden_size"]),
            device=self.main_device,
            dtype=self.model.dtype,
        )
        source_names = [None] * self.config["n_src"]
        curr_src_idx = 0
        def append_source(src_tensor: torch.Tensor, name: str):
            nonlocal curr_src_idx
            src_buffer[:, :, curr_src_idx, :].copy_(src_tensor.to(self.main_device))
            source_names[curr_src_idx] = name
            curr_src_idx += 1

        append_source(d0, "blocks.0.hook_resid_pre")

        for layer in tqdm(range(self.config["n_layers"]), desc="Scoring"):
            src_view = src_buffer[:, :, :curr_src_idx, :]

            head_dests = [
                f"blocks.{layer}.hook_q_input",
                f"blocks.{layer}.hook_k_input",
                f"blocks.{layer}.hook_v_input",
            ]

            # 2. Attention Head Attribution: Score S -> Each Head
            for dest_name in head_dests:
                if dest_name not in clean_grads:
                    raise ValueError(f"Missing required gradient for {dest_name} to compute attention head scores. Check if backward hooks are registered correctly and gradients are being captured.")

                grad = clean_grads[dest_name].to(self.main_device)
                score_tensor = torch.einsum("bsid,bshd->bshi", src_view, grad)
                if not return_per_token_scores:
                    score_tensor = score_tensor.sum(dim=(0, 1))
                scores[dest_name] = score_tensor.detach().cpu()

            # 3. Source: Attention Output
            res_name = f"blocks.{layer}.attn.hook_result"
            if res_name not in activation_differences:
                raise ValueError(f"Missing required activation difference for {res_name} to compute attention output contributions. Check if forward hooks are registered correctly and activations are being captured.")
 
            res_diff = activation_differences[res_name]
            n_new = res_diff.shape[2]
            src_buffer[:, :, curr_src_idx:curr_src_idx + n_new, :].copy_(res_diff.to(self.main_device))
            for h_idx in range(n_new):
                source_names[curr_src_idx + h_idx] = f"blocks.{layer}.attn.hook_result[{h_idx}]"
            curr_src_idx += n_new

            # 4. MLP Attribution: Score S -> MLP
            src_view = src_buffer[:, :, :curr_src_idx, :]
            mlp_dest_name = f"blocks.{layer}.hook_mlp_in"
            if mlp_dest_name not in clean_grads:
                raise ValueError(f"Missing required gradient for {mlp_dest_name} to compute MLP scores. Check if backward hooks are registered correctly and gradients are being captured.")
            grad = clean_grads[mlp_dest_name].to(self.main_device)
            score_tensor = torch.einsum("bsid,bsd->bsi", src_view, grad)
            if not return_per_token_scores:
                score_tensor = score_tensor.sum(dim=(0, 1))
            scores[mlp_dest_name] = score_tensor.detach().cpu()

            # 5. Source: MLP Output
            mlp_out_name = f"blocks.{layer}.hook_mlp_out"
            if not mlp_out_name in activation_differences:
                raise ValueError(f"Missing required activation difference for {mlp_out_name} to compute MLP output contributions. Check if forward hooks are registered correctly and activations are being captured.")
            d = activation_differences[mlp_out_name]
            append_source(d, f"blocks.{layer}.hook_mlp_out")

            # 6. Residual Output Attribution (Final Layer Only): Score S -> Resid Post
            if layer == self.config["n_layers"] - 1:
                resid_name = f"blocks.{layer}.hook_resid_post"
                if resid_name not in clean_grads:
                    raise ValueError(f"Missing required gradient for {resid_name} to compute final residual stream scores. Check if backward hooks are registered correctly and gradients are being captured.")
                src_view = src_buffer[:, :, :curr_src_idx, :] # shape [B, S, Src, D]
                grad = clean_grads[resid_name].to(self.main_device) # shape [B, S, D]
                score_tensor = torch.einsum("bsid,bsd->bsi", src_view, grad)
                if not return_per_token_scores:
                    score_tensor = score_tensor.sum(dim=(0, 1))
                scores[resid_name] = score_tensor.detach().cpu()
        
        scores["source_names"] = source_names
        
        self.profiler.end(profile, phase_start, "scoring", report_peak=True)

        if curr_src_idx != self.config["n_src"]:
            raise ValueError(f"Unexpected number of sources: {curr_src_idx} (max_sources={self.config['n_src']}). Check if all expected activation differences were computed and appended correctly.")

        return scores
    
# %%

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    dtype = torch.bfloat16
    
    # Example Usage
    
    # model_id = "gpt2"
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # chat_format = False
    
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    chat_format = True
    
    print(f"Loading {model_id}...")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%

if __name__ == "__main__":
    profile = True
    use_subject_noise_baseline = True
    return_per_token_scores = True
    
    # If an `eap` instance exists from a previous run, reset its hooks
    # so we don't leak hooks across runs before creating a new instance.
    if "eap" in globals():
        try:
            existing = globals().get("eap")
            if existing is not None:
                existing.reset_hooks()
                print("Existing EAP instance hooks reset.")
        except Exception:
            pass

    eap = EAPGraph(model)
    
    batch_size = 8
    prompts = [
        "Suppose Albert Einstein lives on a houseboat, Ellie Kemper lives in an apartment, and Elvis Presley lives in a cabin. Therefore, the person living in the apartment is a citizen of",
    ] * batch_size
    subjects = [
        "Ellie Kemper",
    ] * batch_size
    corrupted_prompts = [
        "Suppose Albert Einstein lives on a houseboat, Cillian Murphy lives in an apartment, and Elvis Presley lives in a cabin. Therefore, the person living in the apartment is a citizen of",
    ] * batch_size
    
    # Pre-process inputs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    formatted_prompts = format_prompt(tokenizer, prompts, chat_format=chat_format)
    clean, subject_spans = encode_prompt(tokenizer, formatted_prompts, subjects)
    input_ids = clean["input_ids"]
    attention_mask = clean["attention_mask"]
    
    print(f"Computing attributions. input_ids shape: {input_ids.shape}")
    print("Formatted prompts:\n", formatted_prompts[0])

    kl_div_metric = get_kl_div_metric()
    
    if not use_subject_noise_baseline:
        corrupted_formatted_prompts = format_prompt(tokenizer, corrupted_prompts, chat_format=chat_format)
        corrupted, _ = encode_prompt(tokenizer, corrupted_formatted_prompts)
        corrupted_input_ids = corrupted["input_ids"]
        corrupted_attention_mask = corrupted["attention_mask"]
        
        print("Corrupted prompts:\n", corrupted_formatted_prompts[0])
        
        results = eap.attribute(input_ids, attention_mask, kl_div_metric, corrupted_input_ids=corrupted_input_ids, corrupted_attention_mask=corrupted_attention_mask, integrated_gradients=5, return_per_token_scores=return_per_token_scores, profile=profile)
    elif use_subject_noise_baseline:
        print("Using subject noise baseline...")
        results = eap.attribute(input_ids, attention_mask, kl_div_metric, corrupted_input_ids=None, corrupted_attention_mask=None, subject_spans=subject_spans, integrated_gradients=5, return_per_token_scores=return_per_token_scores, profile=profile)
    
# %%

if __name__ == "__main__":
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=20)
    
    print("Generated output:\n", tokenizer.decode(generated[0]))

# %%

if __name__ == "__main__":
    import json
    # Structured analysis outputs for downstream fine-tuning decisions
    top_mlp_hubs_aggregated = find_top_mlp_hubs_aggregated(results, n=5)
    print("Top MLP hubs by aggregation:\n", json.dumps(top_mlp_hubs_aggregated, indent=4))
    
    if return_per_token_scores:
        top_mlp_hubs_by_sample = find_top_mlp_hubs_by_sample(results, n=5)
        top_mlp_hubs_by_token_sample = find_top_mlp_hubs_by_token_sample(results, token_level_n=5)
    
        print("Top MLP hubs by sample:\n", json.dumps(top_mlp_hubs_by_sample, indent=4))
        print("Top MLP hubs by token and sample:\n", json.dumps(top_mlp_hubs_by_token_sample, indent=4))
        start = 32
        print("Top token:\n", tokenizer.decode(input_ids[0][start:]))

 # %%

if __name__ == "__main__":
    top_hubs_aggregated = find_top_hubs_aggregated(results, n=5)
    print("Top hubs by aggregation:\n", json.dumps(top_hubs_aggregated, indent=4))
    
    if return_per_token_scores:
        top_hubs_by_sample = find_top_hubs_by_sample(results, n=5)
        top_token_hubs_by_sample = find_top_hubs_by_token_sample(results, token_level_n=5)
        
        print("Top hubs by sample:\n", json.dumps(top_hubs_by_sample, indent=4))
        print("Top token hubs by sample:\n", json.dumps(top_token_hubs_by_sample, indent=4))

# %%

if __name__ == "__main__":
    
    top_edges_aggregated = find_top_edges_aggregated(results, n=3)
    print("Top edges by aggregation:\n", json.dumps(top_edges_aggregated, indent=4))
    
    if return_per_token_scores:
        top_edges_by_token_sample = find_top_edges_by_token_sample(results, token_level_n=3)
        top_edges_by_sample = find_top_edges_by_sample(results, n=3)
    
        print("Top edge by sample:\n", json.dumps(top_edges_by_sample, indent=4))
        print("Top edges by token and sample:\n", json.dumps(top_edges_by_token_sample, indent=4))

# %%