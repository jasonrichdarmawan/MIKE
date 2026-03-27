# %%

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Callable, Optional
import re

# %%

import logging
logging.basicConfig(level=logging.DEBUG)

# %%

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
        "o": layer.attn.c_proj,
        "fused_qkv": layer.attn.c_attn,
        "ln_1": layer.ln_1, # Component responsible for Norm before Attention
        "ln_2": layer.ln_2, # Component responsible for Norm before MLP
        "is_fused_qkv": True,
        "is_conv1d": True,
        "mlp_in": layer.mlp.c_fc,
        "mlp_out": layer.mlp.c_proj,
        "layer_block": layer
    }, embed

def get_llama_like_components(model, layer_idx):
    layer = model.model.layers[layer_idx]
    embed = model.model.embed_tokens
    return {
        "o": layer.self_attn.o_proj,
        "q": layer.self_attn.q_proj,
        "k": layer.self_attn.k_proj,
        "v": layer.self_attn.v_proj,
        "ln_1": layer.input_layernorm, # Component responsible for Norm before Attention
        "ln_2": layer.post_attention_layernorm, # MLP Norm
        "is_fused_qkv": False,
        "is_conv1d": False,
        "mlp_in": layer.mlp, # Use full MLP block to capture gradients from both Gate and Up paths
        "mlp_out": layer.mlp.down_proj,
        "layer_block": layer
    }, embed

def get_gptj_components(model, layer_idx):
    layer = model.transformer.h[layer_idx]
    embed = model.transformer.wte
    return {
        "o": layer.attn.out_proj,
        "q": layer.attn.q_proj,
        "k": layer.attn.k_proj,
        "v": layer.attn.v_proj,
        "ln_1": layer.ln_1, # Component responsible for Norm before Attention
        "ln_2": layer.ln_1, # Shared Norm for MLP in parallel architecture
        "is_fused_qkv": False,
        "is_conv1d": False,
        "mlp_in": layer.mlp.fc_in,
        "mlp_out": layer.mlp.fc_out,
        "layer_block": layer
    }, embed

COMPONENT_REGISTRY = {
    "gpt2": get_gpt2_components,
    "llama_like": get_llama_like_components,
    "gptj": get_gptj_components
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
    def __init__(self, model, graph_type=None):
        """
        model: Pre-loaded HuggingFace model
        graph_type: Optional string ('gpt2', 'llama_like', 'gptj'). If None, mostly auto-detected.
        """
        self.model = model
        
        if hasattr(self.model, "hf_device_map"):
             # For sharded models, main_device is a bit murky, but we need *some* device for tensors
             d = list(self.model.hf_device_map.values())[0]
             # Handle string outputs like 'cpu' or 'cuda:0'
             if isinstance(d, str):
                  self.main_device = torch.device(d)
             else:
                  self.main_device = d
        else:
             self.main_device = self.model.device
        
        # Arch detection
        if graph_type:
            self.arch_type = graph_type
        else:
            self.arch_type = self._detect_architecture(self.model)

        if self.arch_type not in COMPONENT_REGISTRY:
            raise ValueError(f"Architecture '{self.arch_type}' not supported or registered. "
                             f"Available: {list(COMPONENT_REGISTRY.keys())}")

        self.config = self.model.config
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        
        if hasattr(self.config, "num_key_value_heads"):
            self.n_kv_heads = self.config.num_key_value_heads
        else:
            self.n_kv_heads = self.n_heads
            
        print(f"Initialized EAPGraph for {self.arch_type}. Layers: {self.n_layers}, Heads: {self.n_heads}")
        
        # Storage
        self.activations = {}
        self.gradients = {}
        self.norm_io = {}
        self.handles = []

    def _detect_architecture(self, model):
        name = model.__class__.__name__.lower()
        if "gpt2" in name: return "gpt2"
        if "llama" in name or "qwen" in name: return "llama_like"
        if "gptj" in name: return "gptj"
        return "unknown"

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
        grad_out = grad_out.to(norm_output.device).to(norm_output.dtype)

        if grad_out.dim() == 3:
            return torch.autograd.grad(
                outputs=norm_output,
                inputs=norm_input,
                grad_outputs=grad_out,
                retain_graph=True,
                allow_unused=False,
            )[0]

        if grad_out.dim() == 4:
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
    def get_activation_hook(self, name, type="std", layer_idx=None):
        def hook(module, input, output=None): 
            # Pre-hooks: (module, input) -> None/Modified Input
            # Post-hooks: (module, input, output) -> None/Modified Output
            
            if type == "resid_pre":
                self.activations[name] = input[0].detach().cpu()

            elif type == "resid_mid":
                self.activations[name] = input[0].detach().cpu()

            elif type == "norm_io":
                self.norm_io[name] = {
                    "input": input[0],
                    "output": output,
                }
                
            elif type == "mlp_out":
                self.activations[name] = output.detach().cpu()
                
            elif type == "attn.result":
                # Pre-hook on O_proj. Input is (attn_output,).
                # We need separate head contributions [B, S, H, D_model].
                # The normal O_proj output sums these [B, S, D_model], so we must
                # manually multiply input heads by their weight slices to keep them separate.
                x = input[0].detach() 
                b, s, _ = x.shape
                
                comps, _ = self._get_layer_components(layer_idx)
                # Note: getting comps inside hook is slightly inefficient but safe for registry access
                w_o = comps["o"].weight 
                
                # Reshape x to [B, S, H, D]
                x_heads = x.view(b, s, self.n_heads, self.head_dim)
                
                if comps["is_conv1d"]:
                    # Conv1D: Y = X @ W + b. W is [Hidden, Hidden] -> [In, Out]
                    w_reshaped = w_o.view(self.n_heads, self.head_dim, self.hidden_size)
                    results = torch.einsum("bshd,hdo->bsho", x_heads, w_reshaped)
                else: 
                    # Linear: Y = X @ W.T. W is [Out, In] -> [Hidden, Hidden]
                    # Note: w_o is [Hidden, Hidden] for 'o_proj'.
                    # For Llama o_proj[Out, In] -> [Hidden, Hidden]. In=Hidden, Out=Hidden.
                    w_reshaped = w_o.view(self.hidden_size, self.n_heads, self.head_dim)
                    results = torch.einsum("bshd,ohd->bsho", x_heads, w_reshaped)
                    
                self.activations[name] = results.detach().cpu()
                
            else:
                self.activations[name] = output.detach().cpu()
        return hook

    # --- Gradient Hooks ---
    def get_gradient_hook(self, name, type="std", layer_idx=None):
        """
        Retrieves gradients for attention heads and MLPs.
        
        For Attention/MLP destinations, this function backpropagates through normalization
        using autograd VJPs from cached norm module input/output tensors, which keeps
        attribution model-agnostic across LayerNorm/RMSNorm variants.
        
        This enables proper attribution of previous layer outputs -> current layer heads.
        """
        def hook(module, grad_input, grad_output):
            if type == "qkv_input":
                # Fused Case (GPT-2)
                comps, _ = self._get_layer_components(layer_idx)
                gy = grad_output[0].detach() 
                w = comps["fused_qkv"].weight 
                
                split_size = self.hidden_size
                g_q, g_k, g_v = gy.split(split_size, dim=-1)
                
                b, s, _ = g_q.shape
                
                if comps["is_conv1d"]:
                    w_q, w_k, w_v = w.split(split_size, dim=1)
                    
                    g_q_heads = g_q.view(b, s, self.n_heads, self.head_dim)
                    g_k_heads = g_k.view(b, s, self.n_kv_heads, self.head_dim)
                    g_v_heads = g_v.view(b, s, self.n_kv_heads, self.head_dim)
                    
                    w_q_heads = w_q.view(self.hidden_size, self.n_heads, self.head_dim)
                    w_k_heads = w_k.view(self.hidden_size, self.n_kv_heads, self.head_dim)
                    w_v_heads = w_v.view(self.hidden_size, self.n_kv_heads, self.head_dim)
                    
                    # einsum: b s h d_out, d_in h d_out -> b s h d_in
                    gx_q = torch.einsum("bsho,iho->bshi", g_q_heads, w_q_heads)
                    gx_k = torch.einsum("bsho,iho->bshi", g_k_heads, w_k_heads)
                    gx_v = torch.einsum("bsho,iho->bshi", g_v_heads, w_v_heads)
                else:
                    raise NotImplementedError("Fused QKV non-Conv1D path is not implemented.")
                
                norm_key = f"blocks.{layer_idx}.ln_1"
                gx_q = self._norm_vjp(norm_key, gx_q)
                gx_k = self._norm_vjp(norm_key, gx_k)
                gx_v = self._norm_vjp(norm_key, gx_v)

                self.gradients[f"blocks.{layer_idx}.hook_q_input"] = gx_q.detach().cpu()
                self.gradients[f"blocks.{layer_idx}.hook_k_input"] = gx_k.detach().cpu()
                self.gradients[f"blocks.{layer_idx}.hook_v_input"] = gx_v.detach().cpu()

            elif type in ["q_input", "k_input", "v_input"]:
                # Separate Q/K/V Case (Llama/GPT-J)
                # Why not use grad_input[0]?
                # grad_input[0] is the gradient w.r.t. the input of the linear layer (dL/dX).
                # However, this gradient is SUMMED over all heads (because X is shared).
                # For EAP, we need the gradient w.r.t. X attributed to *specific heads* 
                # (partial L / partial X_h) to compute edge scores S -> Head_h.
                # We can recover this by taking grad_output (dL/dQ_all), splitting it by head, 
                # and backpropagating through that head's specific weight slice.
                comps, _ = self._get_layer_components(layer_idx)
                
                gy = grad_output[0].detach()
                w = module.weight 
                
                b, s, out_dim = gy.shape
                
                if type == "q_input":
                    n_heads = self.n_heads
                else:
                    n_heads = self.n_kv_heads
                
                # View gradients: [B, S, H, D]
                gy_heads = gy.view(b, s, n_heads, self.head_dim)
                
                # View weights: [Out, In] -> [H, D, In] (Transposed logic handled in einsum)
                # Linear weights are (Out, In). Llama/GPTJ usually [H*D, Hidden]
                w_heads = w.view(n_heads, self.head_dim, self.hidden_size)
                
                # Compute gradient w.r.t input of Linear (Post-LN)
                # dL/dX_h = dL/dY_h @ W_h
                gx = torch.einsum("bsho,hoi->bshi", gy_heads, w_heads)
                
                gx = self._norm_vjp(f"blocks.{layer_idx}.ln_1", gx)
                
                self.gradients[name] = gx.detach().cpu()

            elif type == "mlp_in":
                # We want the gradient with respect to the input of the MLP block (in residual stream space).
                # grad_input[0] corresponds to the input tensor (d_model), whereas grad_output 
                # would be the expanded hidden state (d_mlp), which doesn't match the source dimension.
                if grad_input[0] is not None:
                    gx = grad_input[0].detach()

                    norm_key = f"blocks.{layer_idx}.ln_2"
                    if norm_key in self.norm_io:
                        gx = self._norm_vjp(norm_key, gx)
                    
                    self.gradients[name] = gx.detach().cpu()
            
            elif type == "resid_post":
                if grad_output[0] is not None:
                     self.gradients[name] = grad_output[0].detach().cpu()

        return hook

    def register_forward_hooks(self):
        self.reset_hooks()
        for i in range(self.n_layers):
            comps, _ = self._get_layer_components(i)
            # 1. Resid Pre (Block Input)
            self.handles.append(comps["layer_block"].register_forward_pre_hook(
                 self.get_activation_hook(f"blocks.{i}.hook_resid_pre", type="resid_pre")
            ))
            # 2. Head Results (Source) - Hook Input to O_proj
            self.handles.append(comps["o"].register_forward_pre_hook(
                self.get_activation_hook(f"blocks.{i}.attn.hook_result", type="attn.result", layer_idx=i)
            ))

            self.handles.append(comps["ln_1"].register_forward_hook(
                self.get_activation_hook(f"blocks.{i}.ln_1", type="norm_io")
            ))

            # 3. MLP Pre-LN Input (Resid Mid)
            # Necessary for accurate MLP attribution w/ manual LN backprop
            if "ln_2" in comps and comps["ln_2"] is not None:
                self.handles.append(comps["ln_2"].register_forward_hook(
                    self.get_activation_hook(f"blocks.{i}.ln_2", type="norm_io")
                ))

            # 4. MLP Out (Source)
            self.handles.append(comps["mlp_out"].register_forward_hook(
                self.get_activation_hook(f"blocks.{i}.hook_mlp_out", type="mlp_out")
            ))

    def register_backward_hooks(self):
        for i in range(self.n_layers):
            comps, _ = self._get_layer_components(i)
            # 1. Q, K, V Inputs
            if comps["is_fused_qkv"]:
                 self.handles.append(comps["fused_qkv"].register_full_backward_hook(
                     self.get_gradient_hook("dummy", type="qkv_input", layer_idx=i)
                 ))
            else:
                 self.handles.append(comps["q"].register_full_backward_hook(
                     self.get_gradient_hook(f"blocks.{i}.hook_q_input", type="q_input", layer_idx=i)
                 ))
                 self.handles.append(comps["k"].register_full_backward_hook(
                     self.get_gradient_hook(f"blocks.{i}.hook_k_input", type="k_input", layer_idx=i)
                 ))
                 self.handles.append(comps["v"].register_full_backward_hook(
                     self.get_gradient_hook(f"blocks.{i}.hook_v_input", type="v_input", layer_idx=i)
                 ))

            # 2. MLP Input
            self.handles.append(comps["mlp_in"].register_full_backward_hook(
                self.get_gradient_hook(f"blocks.{i}.hook_mlp_in", type="mlp_in", layer_idx=i)
            ))
            # 3. Resid Post
            self.handles.append(comps["layer_block"].register_full_backward_hook(
                self.get_gradient_hook(f"blocks.{i}.hook_resid_post", type="resid_post", layer_idx=i)
            ))

    def _prepare_activations(self, input_ids, attention_mask, corrupted_input_ids):
        if corrupted_input_ids is None:
            raise ValueError("corrupted_input_ids is required. Noise-based corruption is no longer supported.")

        self.model.eval()
        self.model.zero_grad()
        self.reset_hooks()

        input_ids = input_ids.to(self.main_device)
        attention_mask = attention_mask.to(self.main_device)
        input_length = attention_mask.sum(dim=-1)
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

        embed_acts = []
        h = embed.register_forward_hook(stat_hook)
        self.register_forward_hooks()
        with torch.no_grad():
            corrupted_logits = self.model(corrupted_input_ids).logits.detach()
            corrupt_acts = self.activations.copy()
        h.remove()
        self.reset_hooks()

        corrupted_wte = embed_acts[0]
        if corrupted_wte.shape != clean_wte.shape:
            raise ValueError(f"Shape mismatch: Clean {clean_wte.shape} vs Corrupt {corrupted_wte.shape}")

        activation_differences = {}
        for name, clean_act in clean_acts.items():
            if name in corrupt_acts:
                activation_differences[name] = corrupt_acts[name] - clean_act

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

    def _compute_gradients(self, input_ids, input_length, clean_wte, corrupted_wte, metric_fn: Callable, reference_logits, steps: int):
        _, embed = self._get_layer_components(0)
        avg_grads = {}

        for step in range(1, steps + 1):
            self.reset_hooks()
            self.model.zero_grad()
            self.register_forward_hooks()
            self.register_backward_hooks()

            def interpolation_hook(module, input, output):
                interpolated = corrupted_wte + (step / steps) * (clean_wte - corrupted_wte)
                interpolated.requires_grad_(True)
                return interpolated

            h_interp = embed.register_forward_hook(interpolation_hook)

            logits = self.model(input_ids).logits
            loss = metric_fn(logits, reference_logits, input_length)
            loss.backward()
            h_interp.remove()

            for name, grad in self.gradients.items():
                if name not in avg_grads:
                    avg_grads[name] = grad.clone()
                else:
                    avg_grads[name] += grad

        for name in avg_grads:
            avg_grads[name] /= steps

        return avg_grads

    def get_activations(self, input_ids, attention_mask, corrupted_input_ids, metric_fn: Callable):
        prepared = self._prepare_activations(input_ids, attention_mask, corrupted_input_ids)
        gradients = self._compute_gradients(
            prepared["input_ids"],
            prepared["input_length"],
            prepared["clean_wte"],
            prepared["corrupted_wte"],
            metric_fn,
            prepared["corrupted_logits"],
            steps=1,
        )
        return prepared["activation_differences"], gradients

    def get_activations_ig(self, input_ids, attention_mask, corrupted_input_ids, metric_fn: Callable, steps: int = 30):
        if steps <= 0:
            raise ValueError(f"integrated_gradients must be > 0, got {steps}")

        prepared = self._prepare_activations(input_ids, attention_mask, corrupted_input_ids)
        gradients = self._compute_gradients(
            prepared["input_ids"],
            prepared["input_length"],
            prepared["clean_wte"],
            prepared["corrupted_wte"],
            metric_fn,
            prepared["clean_logits"],
            steps=steps,
        )
        return prepared["activation_differences"], gradients

    def attribute(
        self,
        input_ids,
        attention_mask,
        metric_fn: Callable,
        integrated_gradients: Optional[int] = None,
        corrupted_input_ids: Optional[torch.Tensor] = None,
        return_per_token_scores: bool = False,
    ):
        """
        Core EAP/EAP-IG execution using pre-processed tensors.
        input_ids: [Batch, Seq]
        attention_mask: [Batch, Seq]
        metric_fn: Callable that maps (logits, reference_logits, input_length) -> scalar metric/loss.
        integrated_gradients: If None, run EAP. If int>0, run EAP-IG with this many steps.
        corrupted_input_ids: [Batch, Seq] corrupted baseline inputs.
        return_per_token_scores: if True, include per-sample/per-position edge scores.
        """
        if integrated_gradients is None:
            activation_differences, clean_grads = self.get_activations(
                input_ids, attention_mask, corrupted_input_ids, metric_fn
            )
        else:
            activation_differences, clean_grads = self.get_activations_ig(
                input_ids, attention_mask, corrupted_input_ids, metric_fn, steps=integrated_gradients
            )

        attention_mask = attention_mask.to(self.main_device)
        mask = attention_mask.unsqueeze(-1).float().cpu()

        if "blocks.0.hook_resid_pre" not in activation_differences:
            raise ValueError("Missing required activation difference for blocks.0.hook_resid_pre")

        scores = {}
        d0 = activation_differences["blocks.0.hook_resid_pre"] * mask

        bsz, seq_len, d_model = d0.shape
        max_sources = 1 + (self.n_layers * self.n_heads) + self.n_layers
        src_buffer = torch.empty(
            (bsz, seq_len, max_sources, d_model),
            device=self.main_device,
            dtype=d0.dtype,
        )
        source_names = [None] * max_sources
        n_src = 0

        def append_source(src_tensor_cpu: torch.Tensor, name: str):
            nonlocal n_src
            src_buffer[:, :, n_src, :].copy_(src_tensor_cpu.to(self.main_device))
            source_names[n_src] = name
            n_src += 1

        append_source(d0, "blocks.0.hook_resid_pre")

        for layer in tqdm(range(self.n_layers), desc="Scoring"):
            src_view = src_buffer[:, :, :n_src, :]

            head_dests = [
                f"blocks.{layer}.hook_q_input",
                f"blocks.{layer}.hook_k_input",
                f"blocks.{layer}.hook_v_input",
            ]

            for dest_name in head_dests:
                if dest_name not in clean_grads:
                    continue

                grad = clean_grads[dest_name].to(self.main_device)
                score_tensor = torch.einsum("bsid,bshd->bshi", src_view, grad)
                if return_per_token_scores:
                    scores[dest_name] = score_tensor.permute(0, 1, 3, 2).detach().cpu()
                else:
                    score_matrix = score_tensor.sum(dim=(0, 1))
                    scores[dest_name] = score_matrix.T.detach().cpu()

            res_name = f"blocks.{layer}.attn.hook_result"
            if res_name in activation_differences:
                res_diff = activation_differences[res_name] * mask.unsqueeze(2)
                n_new = res_diff.shape[2]
                src_buffer[:, :, n_src:n_src + n_new, :].copy_(res_diff.to(self.main_device))
                for h_idx in range(n_new):
                    source_names[n_src + h_idx] = f"blocks.{layer}.attn.hook_result[{h_idx}]"
                n_src += n_new

            src_view = src_buffer[:, :, :n_src, :]
            mlp_dest_name = f"blocks.{layer}.hook_mlp_in"
            if mlp_dest_name in clean_grads:
                grad = clean_grads[mlp_dest_name].to(self.main_device)
                score_tensor = torch.einsum("bsid,bsd->bsi", src_view, grad)
                if return_per_token_scores:
                    scores[mlp_dest_name] = score_tensor.detach().cpu()
                else:
                    score_vec = score_tensor.sum(dim=(0, 1))
                    scores[mlp_dest_name] = score_vec.detach().cpu()

            mlp_out_name = f"blocks.{layer}.hook_mlp_out"
            if mlp_out_name in activation_differences:
                d = activation_differences[mlp_out_name] * mask
                append_source(d, f"blocks.{layer}.hook_mlp_out")

            if layer == self.n_layers - 1:
                resid_name = f"blocks.{layer}.hook_resid_post"
                if resid_name in clean_grads:
                    src_view = src_buffer[:, :, :n_src, :] # shape [B, S, Src, D]
                    grad = clean_grads[resid_name].to(self.main_device) # shape [B, S, D]
                    score_tensor = torch.einsum("bsid,bsd->bsi", src_view, grad)
                    if return_per_token_scores:
                        scores[resid_name] = score_tensor.detach().cpu()
                    else:
                        score_vec = score_tensor.sum(dim=(0, 1))
                        scores[resid_name] = score_vec.detach().cpu()

        scores["source_names"] = source_names[:n_src]
        return scores
    
# %%

if __name__ == "__main__":
    # Example Usage
    model_id = "gpt2"
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
# %%

if __name__ == "__main__":
    # EAP Wrapper
    eap = EAPGraph(model)
    
    prompts = [
        "Suppose John lives on a houseboat, Ellie Kemper lives in an apartment, and Cillian Murphy lives in a cabin. Therefore, the person living in the apartment is a citizen of",
    ]
    corrupted_prompts = [
        "Suppose John lives on a houseboat, Cillian Murphy lives in an apartment, and Ellie Kemper lives in a cabin. Therefore, the person living in the apartment is a citizen of",
    ]
    
    # Pre-process inputs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenized = tokenizer(prompts + corrupted_prompts, return_tensors="pt", padding="longest")

    batch_size = len(prompts)
    input_ids = tokenized["input_ids"][:batch_size]
    attention_mask = tokenized["attention_mask"][:batch_size]
    corrupted_input_ids = tokenized["input_ids"][batch_size:]

    kl_div_metric = get_kl_div_metric()

    return_per_token_scores = True
    results = eap.attribute(input_ids, attention_mask, kl_div_metric, corrupted_input_ids=corrupted_input_ids, integrated_gradients=5, return_per_token_scores=return_per_token_scores)
    
# %%

if __name__ == "__main__":
    import json
    # Structured analysis outputs for downstream fine-tuning decisions
    mlp_hubs_aggregated = find_top_mlp_hubs_aggregated(results, n=5)
    print("MLP hubs aggregated:\n", json.dumps(mlp_hubs_aggregated, indent=4))
    
    if return_per_token_scores:
        mlp_hubs_by_sample = find_top_mlp_hubs_by_sample(results, n=5)
        mlp_hubs_token_by_sample = find_top_mlp_hubs_by_token_sample(results, token_level_n=5)
    
        print("MLP hubs by sample:\n", json.dumps(mlp_hubs_by_sample, indent=4))
        print("MLP hub token by sample:\n", json.dumps(mlp_hubs_token_by_sample, indent=4))

# %%

if __name__ == "__main__":
    
    top_edges_aggregated = find_top_edges_aggregated(results, n=3)
    top_edges_by_sample = find_top_edges_by_sample(results, n=3)
    top_token_edges_by_sample = find_top_edges_by_token_sample(results, token_level_n=3)
    
    print("Top aggregated edge candidates:\n", json.dumps(top_edges_aggregated, indent=4))
    print("Top edge candidates by sample:\n", json.dumps(top_edges_by_sample, indent=4))
    print("Top token edges by sample:\n", json.dumps(top_token_edges_by_sample, indent=4))

# %%