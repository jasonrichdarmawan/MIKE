# %%

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Callable, Optional
import re

# %%

# --- Standalone Utility Functions ---

def find_peak_mlp_hubs(scores, n_layers=None):
    """
    Identifies MLP layers where the total attribution is higher than both
    the previous and next layers (local maxima).
    """
    print("\n--- MLP Hub Analysis ---")
    
    # Infer n_layers if not provided, assuming standard keys
    if n_layers is None:
        # Try to find max layer index from keys
        max_layer = -1
        for k in scores.keys():
            match = re.search(r"blocks\.(\d+)\.", k)
            if match:
                max_layer = max(max_layer, int(match.group(1)))
        n_layers = max_layer + 1

    mlp_sums = []
    for i in range(n_layers):
        key = f"blocks.{i}.hook_mlp_in"
        if key in scores:
            # Sum absolute values across all sources (magnitude-based)
            val = scores[key].abs().sum().item()
            mlp_sums.append(val)
        else:
            mlp_sums.append(0.0)
    
    peaks = []
    # Strictly applies to internal layers (1 to N-2)
    for i in range(1, len(mlp_sums) - 1): 
        curr = mlp_sums[i]
        prev = mlp_sums[i-1]
        next = mlp_sums[i+1]
        
        if curr > prev and curr > next:
            peaks.append((i, curr))
            print(f"Peak at Layer {i} (Score: {curr:.4f}) | Prev: {prev:.4f}, Next: {next:.4f}")
    
    if not peaks:
        print("No internal peaks found satisfying condition.")
        
    return peaks

def print_top_edges(scores, n=30):
    """
    Prints the top N edges by absolute attribution score.
    """
    # Flatten the Tensor-based dict to edges
    flat_scores = {}
    source_names = scores.get("source_names", [])
    
    if not source_names:
        print("Warning: 'source_names' not found in scores. cannot map indices to names.")
        return

    for dest, matrix in scores.items():
        if dest == "source_names": continue
        
        # Matrix is Tensor.
        # If 2D [Head, Src]
        if matrix.dim() == 2:
            n_heads, n_srcs = matrix.shape
            for h in range(n_heads):
                for s in range(n_srcs):
                    if s < len(source_names):
                        src_name = source_names[s]
                        val = matrix[h, s].abs().item()
                        flat_scores[f"{src_name} -> {dest}.H{h}"] = val
                    
        # If 1D [Src]
        elif matrix.dim() == 1:
            n_srcs = matrix.shape[0]
            for s in range(n_srcs):
                if s < len(source_names):
                    src_name = source_names[s]
                    val = matrix[s].abs().item()
                    flat_scores[f"{src_name} -> {dest}"] = val

    # Sort by absolute value
    sorted_edges = sorted(flat_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n--- Top {n} Edges ---")
    for i, (k, v) in enumerate(sorted_edges[:n]):
        # Beautify formatting
        k = k.replace("blocks.", "L")
        k = k.replace(".hook_", ".")
        k = k.replace("attn.result", "H")
        k = k.replace("mlp_out", "MLP")
        k = k.replace("mlp_in", "MLP")
        k = k.replace("q_input", "Q")
        k = k.replace("k_input", "K")
        k = k.replace("v_input", "V")
        k = k.replace("resid_pre", "ResidPre")
        k = k.replace("resid_post", "ResidPost")   
        k = k.replace("[", "").replace("]", "")
        
        print(f"{i+1:02d}. {k}  (Score: {v:.4f})")


def layer_norm_backward_independent(grad_y, x, ln_module):
    """
    Backpropagate through LayerNorm for a specific 'head' slice acting as if independent.
    
    Equations:
    Forward: y = (x - mu) / sigma * gamma + beta
    Let x_hat = (x - mu) / sigma
    dy = grad_y * gamma
    
    Standard LN Backward (dx):
    dx_hat = dy
    dx = (1/sigma) * [ dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat) ]
    
    This function applies the backward pass assuming grad_y is the gradient 
    w.r.t the output of the LayerNorm corresponding to a specific head's 
    downstream usage, properly accounting for the shared normalization statistics.
    """
    x = x.to(grad_y.device).to(grad_y.dtype)
    dim = -1
    
    # 1. Statistics
    mu = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, unbiased=False, keepdim=True)
    std = (var + ln_module.eps).sqrt()
    x_hat = (x - mu) / std
    
    # 2. Gradient of normalized input (dy * gamma)
    gamma = ln_module.weight.to(grad_y.device).to(grad_y.dtype)
    
    # Handle broadcasting for Head-Dimension if present
    # grad_y can be [B, S, H, D] (Attention) or [B, S, D] (MLP)
    if grad_y.dim() == 4:
        # Attention Case: [B, S, H, D]
        # Expand x_hat and std for broadcasting
        x_hat = x_hat.unsqueeze(2) # [B, S, 1, D]
        std = std.unsqueeze(2)     # [B, S, 1, D]
    elif grad_y.dim() == 3:
        # MLP Case: [B, S, D]
        # No extra unsqueeze needed for H dimension
        pass
    
    dx_hat = grad_y * gamma    # [B, S, H, D] or [B, S, D]
    
    # 3. Reduction Terms (averaging over D dimension)
    term2 = dx_hat.mean(dim=dim, keepdim=True) # mean(dy)
    term3 = x_hat * (dx_hat * x_hat).mean(dim=dim, keepdim=True) # x_hat * mean(dy * x_hat)
    
    # 4. Final Gradient (distributed to inputs)
    grad_x = (dx_hat - term2 - term3) / std
    return grad_x

def rms_norm_backward_independent(grad_y, x, rms_module):
    """
    Backpropagate through RMSNorm for a specific 'head' slice.
    
    Equations:
    Forward: y = x * w * rsqrt(mean(x^2) + eps)
    Let r = rsqrt(mean(x^2) + eps)
    y = x * w * r
    
    Backward:
    dx = r * (dy * w - x * mean(dy * w * x) * r^2)
       = r * (dy * w - x * mean(dy * w * x) / (mean(x^2) + eps))
    
    This matches the standard RMSNorm gradient derivation.
    """
    x = x.to(grad_y.device).to(grad_y.dtype)
    dim = -1
    
    # 1. Statistics
    input_dtype = x.dtype
    x_f32 = x.float()
    
    # Variance here matches mean(x^2)
    ms = x_f32.pow(2).mean(dim, keepdim=True)
    r = torch.rsqrt(ms + rms_module.variance_epsilon) # 1/RMS
    
    # 2. Pre-computations
    w = rms_module.weight.float().to(grad_y.device)
    
    # grad_y is [B, S, H, D]
    # w is [D] or [H, D] depending on how it's stored, usually [D] for the full layer.
    # However, for 'independent' backward, we are effectively saying that grad_y 
    # corresponds to a specific slice of the output.
    
    # CAUTION: 'w' in RMSNorm is applying to the *output* of the normalization 
    # (elementwise affine).
    
    # If we are doing "block-independent" attribution, we assume linearity.
    # The term `dy * w` : 
    dy_scaled = grad_y.to(torch.float32) * w
    
    # 3. Reduction Term
    # term = mean(dy_scaled * x)
    # We must be careful with broadcasting.
    # x is [B, S, D], grad_y is [B, S, H, D].
    
    x_expanded = x_f32
    r_expanded = r
    
    if grad_y.dim() == 4:
        # Attention Case: [B, S, H, D]
        x_expanded = x_f32.unsqueeze(2)     # [B, S, 1, D]
        r_expanded = r.unsqueeze(2)         # [B, S, 1, 1]
    
    # term: [B, S, H, 1] OR [B, S, 1]
    term = (dy_scaled * x_expanded).mean(dim=dim, keepdim=True)
    
    # 4. Final Gradient
    # dx = r * (dy_scaled - x * term * r^2)
    # Note: r^2 = 1 / (ms + eps)
    
    ms_expanded = ms
    if grad_y.dim() == 4:
         ms_expanded = ms.unsqueeze(2)

    dx = r_expanded * (dy_scaled - x_expanded * term / (ms_expanded + rms_module.variance_epsilon))
    
    return dx.to(input_dtype)

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

    # --- Activation Hooks ---
    def get_activation_hook(self, name, type="std", layer_idx=None):
        def hook(module, input, output=None): 
            # Pre-hooks: (module, input) -> None/Modified Input
            # Post-hooks: (module, input, output) -> None/Modified Output
            
            if type == "resid_pre":
                self.activations[name] = input[0].detach().cpu()

            elif type == "resid_mid":
                self.activations[name] = input[0].detach().cpu()
                
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
        
        Crucially, for Attention Heads, this function now manually backpropagates through 
        the LayerNorm/RMSNorm (LN1) to attribute gradients to the *Residual Stream* (Pre-LN).
        
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
                
                # --- Manual Norm Backward ---
                ln_module = comps["ln_1"]
                # Must have captured activations previously
                x_pre_ln = self.activations[f"blocks.{layer_idx}.hook_resid_pre"].to(gx_q.device)
                
                if "RMSNorm" in ln_module.__class__.__name__:
                    gx_q = rms_norm_backward_independent(gx_q, x_pre_ln, ln_module)
                    gx_k = rms_norm_backward_independent(gx_k, x_pre_ln, ln_module)
                    gx_v = rms_norm_backward_independent(gx_v, x_pre_ln, ln_module)
                else: 
                    # Assume LayerNorm
                    gx_q = layer_norm_backward_independent(gx_q, x_pre_ln, ln_module)
                    gx_k = layer_norm_backward_independent(gx_k, x_pre_ln, ln_module)
                    gx_v = layer_norm_backward_independent(gx_v, x_pre_ln, ln_module)

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
                
                # --- Manual Norm Backward ---
                ln_module = comps["ln_1"]
                x_pre_ln = self.activations[f"blocks.{layer_idx}.hook_resid_pre"].to(gx.device)
                
                if "RMSNorm" in ln_module.__class__.__name__:
                    gx = rms_norm_backward_independent(gx, x_pre_ln, ln_module)
                else:
                    gx = layer_norm_backward_independent(gx, x_pre_ln, ln_module)
                
                self.gradients[name] = gx.detach().cpu()

            elif type == "mlp_in":
                # We want the gradient with respect to the input of the MLP block (in residual stream space).
                # grad_input[0] corresponds to the input tensor (d_model), whereas grad_output 
                # would be the expanded hidden state (d_mlp), which doesn't match the source dimension.
                if grad_input[0] is not None:
                    gx = grad_input[0].detach()
                     
                    # --- Manual Norm Backward ---
                    # We need to know which LN corresponds to the MLP.
                    # In GPT2/GPTJ, it's ln_2. In Llama, it's post_attention_layernorm.
                    comps, _ = self._get_layer_components(layer_idx)
                     
                    # Check if we have an MLP-specific LN in our components getter.
                    # GPT-2 has layer.ln_2. Llama has layer.post_attention_layernorm
                    if "ln_2" in comps:
                        ln_module = comps["ln_2"]
                         
                        x_pre_ln_2 = self.activations[f"blocks.{layer_idx}.hook_resid_mid"].to(gx.device)
                        
                        if "RMSNorm" in ln_module.__class__.__name__:
                            gx = rms_norm_backward_independent(gx, x_pre_ln_2, ln_module)
                        else:
                            gx = layer_norm_backward_independent(gx, x_pre_ln_2, ln_module)
                    
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

            # 3. MLP Pre-LN Input (Resid Mid)
            # Necessary for accurate MLP attribution w/ manual LN backprop
            if "ln_2" in comps and comps["ln_2"] is not None:
                # Use pre-hook on the LN module itself to capture its input
                self.handles.append(comps["ln_2"].register_forward_pre_hook(
                    self.get_activation_hook(f"blocks.{i}.hook_resid_mid", type="resid_mid")
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
    ):
        """
        Core EAP/EAP-IG execution using pre-processed tensors.
        input_ids: [Batch, Seq]
        attention_mask: [Batch, Seq]
        metric_fn: Callable that maps (logits, reference_logits, input_length) -> scalar metric/loss.
        integrated_gradients: If None, run EAP. If int>0, run EAP-IG with this many steps.
        corrupted_input_ids: [Batch, Seq] corrupted baseline inputs.
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
                score_matrix = torch.einsum("bsid,bshd->ih", src_view, grad)
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
                score_vec = torch.einsum("bsid,bsd->i", src_view, grad)
                scores[mlp_dest_name] = score_vec.detach().cpu()

            mlp_out_name = f"blocks.{layer}.hook_mlp_out"
            if mlp_out_name in activation_differences:
                d = activation_differences[mlp_out_name] * mask
                append_source(d, f"blocks.{layer}.hook_mlp_out")

            if layer == self.n_layers - 1:
                resid_name = f"blocks.{layer}.hook_resid_post"
                if resid_name in clean_grads:
                    src_view = src_buffer[:, :, :n_src, :]
                    grad = clean_grads[resid_name].to(self.main_device)
                    score_vec = torch.einsum("bsid,bsd->i", src_view, grad)
                    scores[resid_name] = score_vec.detach().cpu()

        scores["source_names"] = source_names[:n_src]
        return scores

if __name__ == "__main__":
    # Example Usage
    model_id = "gpt2"
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # EAP Wrapper
    eap = EAPGraph(model)
    
    prompts = [
        "Suppose John lives in a houseboat, Ellie Kemper lives in an apartment, and Keanu Reeves lives in a cabin. Therefore, the person who lives in an apartment is a citizen of",
    ]
    corrupted_prompts = [
        "Suppose John lives in a houseboat, Keanu Reeves lives in an apartment, and Ellie Kemper lives in a cabin. Therefore, the person who lives in an apartment is a citizen of",
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

    results = eap.attribute(input_ids, attention_mask, kl_div_metric, corrupted_input_ids=corrupted_input_ids, integrated_gradients=5)
    
    # Standalone analysis
    find_peak_mlp_hubs(results)
    print_top_edges(results)

# %%