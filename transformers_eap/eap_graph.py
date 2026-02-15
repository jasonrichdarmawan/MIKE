import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Callable, Optional
import re

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

# --- Model Component Registries ---

def get_gpt2_components(model, layer_idx):
    layer = model.transformer.h[layer_idx]
    embed = model.transformer.wte
    return {
        "o": layer.attn.c_proj,
        "fused_qkv": layer.attn.c_attn,
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

class EAPGraph:
    def __init__(self, model, graph_type=None):
        """
        model: Pre-loaded HuggingFace model
        graph_type: Optional string ('gpt2', 'llama_like', 'gptj'). If None, mostly auto-detected.
        """
        self.model = model
        
        if hasattr(self.model, "hf_device_map"):
             self.main_device = list(self.model.hf_device_map.values())[0] 
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
        def hook(module, grad_input, grad_output):
            if type == "qkv_input":
                # Fused Case
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
                    
                self.gradients[f"blocks.{layer_idx}.hook_q_input"] = gx_q.detach().cpu()
                self.gradients[f"blocks.{layer_idx}.hook_k_input"] = gx_k.detach().cpu()
                self.gradients[f"blocks.{layer_idx}.hook_v_input"] = gx_v.detach().cpu()

            elif type in ["q_input", "k_input", "v_input"]:
                # Separate Q/K/V Case
                # Why not use grad_input[0]?
                # grad_input[0] is the gradient w.r.t. the input of the linear layer (dL/dX).
                # However, this gradient is SUMMED over all heads (because X is shared).
                # For EAP, we need the gradient w.r.t. X attributed to *specific heads* 
                # (partial L / partial X_h) to compute edge scores S -> Head_h.
                # We can recover this by taking grad_output (dL/dQ_all), splitting it by head, 
                # and backpropagating through that head's specific weight slice.
                
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
                # Linear weights are (Out, In).
                w_heads = w.view(n_heads, self.head_dim, self.hidden_size)
                
                # We compute dL/dX_h = dL/dY_h @ W_h
                # dL/dY corresponds to gy_heads. W_h is the weight slice for head h.
                # Result is the gradient at the input (X) derived purely from the path through head h.
                gx = torch.einsum("bsho,hoi->bshi", gy_heads, w_heads)
                
                self.gradients[name] = gx.detach().cpu()

            elif type == "mlp_in":
                # We want the gradient with respect to the input of the MLP block (in residual stream space).
                # grad_input[0] corresponds to the input tensor (d_model), whereas grad_output 
                # would be the expanded hidden state (d_mlp), which doesn't match the source dimension.
                if grad_input[0] is not None:
                    self.gradients[name] = grad_input[0].detach().cpu()
            
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
            # 3. MLP Out (Source)
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
                self.get_gradient_hook(f"blocks.{i}.hook_mlp_in", type="mlp_in")
            ))
            # 3. Resid Post
            self.handles.append(comps["layer_block"].register_full_backward_hook(
                self.get_gradient_hook(f"blocks.{i}.hook_resid_post", type="resid_post")
            ))

    def run_eap(
        self,
        input_ids,
        attention_mask,
        subject_indices,
        ans_ids=None,
        noise_scale=0.1,
        metric_fn: Optional[Callable] = None,
    ):
        """
        Core EAP execution using pre-processed tensors.
        input_ids: [Batch, Seq]
        attention_mask: [Batch, Seq]
        subject_indices: List of lists of indices for each batch item
        ans_ids: [Batch] Target token IDs (used when metric_fn is None)
        metric_fn: Callable that maps logits -> scalar metric/loss (optional)
        """
        # --- Clean Backward ---
        self.model.eval()
        self.model.zero_grad()
        self.reset_hooks()
        self.register_backward_hooks()
        
        logits = self.model(input_ids).logits
        if metric_fn is not None:
            loss = metric_fn(logits)
        else:
            if ans_ids is None:
                raise ValueError("ans_ids must be provided when metric_fn is None")
            batch_indices = torch.arange(len(input_ids)).to(self.main_device)
            loss = -logits[:, -1, :].log_softmax(dim=-1)[batch_indices, ans_ids].mean()
        print(f"Clean Loss: {loss.item():.4f}")
        loss.backward()
        
        # Collect & CPU offload
        clean_grads = {k: v.clone() for k, v in self.gradients.items()}
        self.reset_hooks()
        self.model.zero_grad()
        
        # --- Clean Forward ---
        self.register_forward_hooks()
        with torch.no_grad():
            self.model(input_ids)
            # We use .copy() to decouple clean_acts from self.activations, which will be 
            # overwritten in the next forward pass. Deep cloning (v.clone()) is not strictly 
            # necessary as hooks create new tensor instances, so shallow copy saves memory/time.
            clean_acts = self.activations.copy()
            
        # --- Noise Stats (from WTE) ---
        _, embed = self._get_layer_components(0) 
        embed_acts = []
        def stat_hook(m, i, o): embed_acts.append(o.detach())
        h = embed.register_forward_hook(stat_hook)
        with torch.no_grad(): self.model(input_ids)
        h.remove()
        clean_wte = embed_acts[0]

        subj_vecs = []
        for b, idxs in enumerate(subject_indices):
            if idxs: subj_vecs.append(clean_wte[b, idxs, :])
        if subj_vecs:
            noise_std = noise_scale * torch.cat(subj_vecs, dim=0).std().item()
        else:
            noise_std = noise_scale * 0.1
        print(f"Noise Std: {noise_std:.4f}")

        # --- Corrupted Forward ---
        def noise_hook(module, input, output):
            noise = torch.randn_like(output) * noise_std
            mask = torch.zeros_like(output, dtype=torch.bool)
            for b, idxs in enumerate(subject_indices):
                mask[b, idxs, :] = True
            return output + (noise * mask)

        h_noise = embed.register_forward_hook(noise_hook)
        with torch.no_grad():
            self.model(input_ids)
            corrupt_acts = self.activations.copy()
        h_noise.remove()
        self.reset_hooks()
        
        # --- Score Computation ---
        scores = {}
        mask = attention_mask.unsqueeze(-1).float().cpu()

        # Initial Source: blocks.0.hook_resid_pre
        d0 = (corrupt_acts["blocks.0.hook_resid_pre"] - clean_acts["blocks.0.hook_resid_pre"]) * mask

        # Pre-allocate a single growing GPU source buffer.
        # Max sources = 1 resid_pre + n_layers * n_heads (attn results) + n_layers (mlp_out)
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
                if dest_name not in clean_grads: continue

                grad = clean_grads[dest_name].to(self.main_device)

                # Einsum: b s src d, b s head d -> src head
                score_matrix = torch.einsum("bsid,bshd->ih", src_view, grad)
                scores[dest_name] = score_matrix.T.detach().cpu()

            # 2. Update Sources (Add Attn Results)
            res_name = f"blocks.{layer}.attn.hook_result"
            if res_name in clean_acts:
                res_diff = (corrupt_acts[res_name] - clean_acts[res_name]) * mask.unsqueeze(2)  # [B,S,H,D] on CPU
                n_new = res_diff.shape[2]
                src_buffer[:, :, n_src:n_src + n_new, :].copy_(
                    res_diff.to(self.main_device)
                )
                for h_idx in range(n_new):
                    source_names[n_src + h_idx] = f"blocks.{layer}.attn.hook_result[{h_idx}]"
                n_src += n_new

            # 3. Score MLP with updated source view
            src_view = src_buffer[:, :, :n_src, :]

            mlp_dest_name = f"blocks.{layer}.hook_mlp_in"
            if mlp_dest_name in clean_grads:
                grad = clean_grads[mlp_dest_name].to(self.main_device)
                # Einsum: b s src d, b s d -> src
                score_vec = torch.einsum("bsid,bsd->i", src_view, grad)
                scores[mlp_dest_name] = score_vec.detach().cpu()

            # 4. Update Sources (Add MLP Out)
            mlp_out_name = f"blocks.{layer}.hook_mlp_out"
            if mlp_out_name in clean_acts:
                d = (corrupt_acts[mlp_out_name] - clean_acts[mlp_out_name]) * mask
                append_source(d, f"blocks.{layer}.hook_mlp_out")

            # 5. Score Resid Post (Last Layer Only)
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
    from transformers_eap.tokenizer import EAPTokenizer # Local import for testing

    model_id = "gpt2"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # EAP Wrapper
    eap = EAPGraph(model)
    eap_tokenizer = EAPTokenizer(tokenizer, device=eap.main_device)
    
    prompts = [
        "The capital of Spain is",
        "The capital of France is", 
        "The capital of Italy is",
        "The capital of Russia is"
    ]
    subjects = ["Spain", "France", "Italy", "Russia"]
    ans_tokens = [" Madrid", " Paris", " Rome", " Moscow"]
    ans_ids = [tokenizer(a)["input_ids"][0] for a in ans_tokens]
    ans_ids = torch.tensor(ans_ids).to(eap.main_device)
    
    # Pre-process inputs
    input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
    
    results = eap.run_eap(input_ids, attention_mask, subject_indices, ans_ids, noise_scale=0.1)
    
    # Standalone analysis
    find_peak_mlp_hubs(results)
    print_top_edges(results)
