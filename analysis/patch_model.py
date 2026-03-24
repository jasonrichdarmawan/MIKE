"""
Model-agnostic neuron attribution patches for GPT-2 and Llama model families.

Applies in-memory monkey-patches to capture internal states needed for neuron-level
LPI analysis — no file replacement, patches live only for the current Python process.

Usage:
    import patch_model
    model = AutoModelForCausalLM.from_pretrained(...)
    patch_model.patch(model)

    with torch.no_grad():
        model(input_ids, use_cache=False)

    states = patch_model.extract_states(layer_num)
    # states[i]: dict with keys —
    #   layer_input, attn_output, residual_output, ffn_output, layer_output,
    #   attn_subvalues, coeff_scores, attn_weights

    info = patch_model.model_info(model)
    # info: dict with ln_f_weight, lm_head, layers,
    #              get_attn_o_proj(layer), get_ffn_down_proj(layer)

Notes:
    - Call model with use_cache=False to avoid conflicts with KV-cache return values.
    - Compatible with transformers 4.30.x through 4.44.2.
"""

import torch

# ── Shared state container ────────────────────────────────────────────────────

_states = {}  # {layer_i: state_dict} — written during forward, read by extract_states()


def extract_states(layer_num):
    """Return list of per-layer state dicts captured in the last forward pass."""
    return [_states[i] for i in range(layer_num)]


# ── Entry point ───────────────────────────────────────────────────────────────

def patch(model):
    """Auto-detect model family and apply attribution patches in-place."""
    _states.clear()
    name = type(model).__name__
    if "GPT2" in name:
        _patch_gpt2(model)
    elif "Llama" in name:
        _patch_llama(model)
    else:
        raise ValueError(
            f"Unsupported model type: {name}. Supported: GPT2LMHeadModel, LlamaForCausalLM."
        )


def model_info(model):
    """
    Return model-agnostic accessors for LPI and query computation.

    Keys:
        ln_f_weight       : final LayerNorm weight [hidden]
        lm_head           : nn.Linear mapping hidden -> vocab
        layers            : list of decoder layer modules
        get_attn_o_proj   : callable(layer) -> [heads, head_dim, hidden] weight tensor
        get_ffn_down_proj : callable(layer) -> [intermediate, hidden] weight tensor
        get_ffn_key       : callable(layer, neuron_id) -> [hidden] input weight for FFN neuron
        get_ffn_ln_weight : callable(layer) -> [hidden] pre-FFN layer norm weight
        get_attn_v_key    : callable(layer, flat_neuron) -> [hidden] input weight for V neuron
                            where flat_neuron = head_i * head_dim + neuron_i
        get_attn_ln_weight: callable(layer) -> [hidden] pre-attention layer norm weight
    """
    name = type(model).__name__
    if "GPT2" in name:
        _n_embd = model.config.n_embd
        return dict(
            ln_f_weight=model.transformer.ln_f.weight.data,
            lm_head=model.lm_head,
            layers=model.transformer.h,
            # Conv1D weight shape: [hidden, hidden] → view as [heads, head_dim, hidden]
            get_attn_o_proj=lambda layer: (
                layer.attn.c_proj.weight.data
                .view(model.config.n_head,
                      model.config.n_embd // model.config.n_head, -1)
            ),
            # Conv1D weight shape: [ffn_dim, hidden] — already correct
            get_ffn_down_proj=lambda layer: layer.mlp.c_proj.weight.data,
            # Conv1D c_fc weight shape: [hidden, ffn_dim]; column n is the key for neuron n
            get_ffn_key=lambda layer, n: layer.mlp.c_fc.weight.data[:, n],
            get_ffn_ln_weight=lambda layer: layer.ln_2.weight.data,
            # Conv1D c_attn weight shape: [hidden, 3*hidden]; V starts at col 2*hidden
            get_attn_v_key=lambda layer, n: layer.attn.c_attn.weight.data[:, _n_embd * 2 + n],
            get_attn_ln_weight=lambda layer: layer.ln_1.weight.data,
        )
    elif "Llama" in name:
        head_num = model.config.num_attention_heads
        head_dim = model.config.hidden_size // head_num
        return dict(
            ln_f_weight=model.model.norm.weight.data,
            lm_head=model.lm_head,
            layers=model.model.layers,
            # Linear weight shape: [hidden, head_num*head_dim] → .T → [head_num*head_dim, hidden]
            # .T is non-contiguous, so use .reshape() (not .view()) to avoid RuntimeError.
            get_attn_o_proj=lambda layer: (
                layer.self_attn.o_proj.weight.data.T
                .reshape(head_num, head_dim, -1)
            ),
            # Linear weight shape: [hidden, intermediate] → .T → [intermediate, hidden]
            get_ffn_down_proj=lambda layer: layer.mlp.down_proj.weight.data.T,
            # Linear gate_proj weight shape: [ffn_dim, hidden]; row n is the key for neuron n
            get_ffn_key=lambda layer, n: layer.mlp.gate_proj.weight.data[n, :],
            get_ffn_ln_weight=lambda layer: layer.post_attention_layernorm.weight.data,
            # Linear v_proj weight shape: [kv_heads*head_dim, hidden]; row n is the key for flat V neuron n
            # Caveat: for GQA models (num_kv_heads != num_heads), flat_neuron indexes the
            # expanded head space. Caller must map head_i -> kv_head_i = head_i // num_groups
            # and recompute flat_neuron = kv_head_i * head_dim + neuron_i before calling.
            get_attn_v_key=lambda layer, n: layer.self_attn.v_proj.weight.data[n, :],
            get_attn_ln_weight=lambda layer: layer.input_layernorm.weight.data,
        )
    else:
        raise ValueError(f"Unsupported model type: {name}")


# ── GPT-2 patches ─────────────────────────────────────────────────────────────

def _patch_gpt2(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2Block

    # Patch 1: GPT2Attention.forward
    _orig_attn_forward = GPT2Attention.forward

    def _patched_attn_forward(self, hidden_states, layer_past=None, attention_mask=None, 
                               head_mask=None, encoder_hidden_states=None,
                               encoder_attention_mask=None, use_cache=False, 
                               output_attentions=False, **kwargs):
        # Recompute value_states from c_attn
        src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        raw = self.c_attn(src)
        splits = raw.split(self.split_size, dim=2)
        value_raw = splits[-1] # [batch, seq_len, hidden]
        bsz, seq_len, _ = value_raw.shape
        value_states = value_raw.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch, heads, seq_len, head_dim]

        # Call original forward
        outputs = _orig_attn_forward(
            self, hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            **kwargs
        )
        # GPT-2 returns (attn_output, attn_weights) when use_cache=False,
        # and (attn_output, present, attn_weights) when use_cache=True.
        # attn_weights is always the last element when output_attentions=True.
        attn_output = outputs[0] # [batch, seq_len, hidden]
        attn_weights = outputs[-1] # [batch, heads, seq_q, seq_k]

        last_weights = attn_weights[:, :, -1, :].unsqueeze(-1) # [batch, heads, seq_k, 1]
        self._last_attn_subvalues = last_weights * value_states # [batch, heads, seq_k, head_dim]
        self._last_attn_weights = attn_weights

        if not output_attentions:
            attn_weights = None

        # Preserve original structure: (attn_output, present, [attn_weights])
        return (attn_output, outputs[1]) + ((attn_weights,) if attn_weights is not None else ())

    GPT2Attention.forward = _patched_attn_forward

    # Patch 2: GPT2MLP.forward
    def _patched_mlp(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        self._last_coefficient_scores = hidden_states  # post-activation: actual coefficient on each value vector
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    GPT2MLP.forward = _patched_mlp

    # Patch 3: GPT2Block.forward
    for layer_i, block in enumerate(model.transformer.h):
        block._layer_i = layer_i

    def _patched_block(self, hidden_states, layer_past=None, attention_mask=None, 
                       head_mask=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, use_cache=False, 
                       output_attentions=False, **kwargs):
        layer_input = hidden_states # [batch, seq_len, hidden]

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            **kwargs,
        ) # returns (attn_output, present, attn_weights)
        attn_output = attn_outputs[0] # [batch, seq_len, hidden]
        
        attn_weights = self.attn._last_attn_weights
        attn_subvalues = self.attn._last_attn_subvalues # [batch, heads, seq_k, head_dim]

        hidden_states = attn_output + residual
        residual_output = hidden_states # [batch, seq_len, hidden]

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=True,
            )
            hidden_states = residual + cross_attn_outputs[0]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ffn_output = self.mlp(hidden_states) # [batch, seq_len, hidden]
        coeff_scores = self.mlp._last_coefficient_scores # [batch, seq_len, ffn_dim]
        hidden_states = residual + ffn_output
        layer_output = hidden_states # [batch, seq_len, hidden]

        _states[self._layer_i] = dict(
            layer_input=layer_input[0], # shape [seq, hidden]
            attn_output=attn_output[0], # shape [seq, hidden]
            residual_output=residual_output[0], # shape [seq, hidden]
            ffn_output=ffn_output[0], # shape [seq, hidden]
            layer_output=layer_output[0], # shape [seq, hidden]
            attn_subvalues=attn_subvalues[0], # shape [heads, seq_k, head_dim]
            coeff_scores=coeff_scores[0], # shape [seq, hidden]
            attn_weights=attn_weights[0], # shape [seq, hidden]
        )

        # Return (hidden_states, presents) etc.
        return (hidden_states,) + attn_outputs[1:]

    GPT2Block.forward = _patched_block




# ── Llama patches ─────────────────────────────────────────────────────────────

def _patch_llama(model):
    """
    Patch LlamaAttention, LlamaMLP, and LlamaDecoderLayer.
    """
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaMLP, LlamaDecoderLayer,
    )

    # Patch 1: LlamaAttention.forward
    _orig_llama_attn = LlamaAttention.forward

    def _patched_llama_attn(self, hidden_states, attention_mask=None, position_ids=None,
                             past_key_value=None, output_attentions=False, use_cache=False,
                             **kwargs):
        # Run original forward with output_attentions=True to obtain attn_weights.
        outputs = _orig_llama_attn(
            self, hidden_states, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output, attn_weights = outputs[0], outputs[1]
        past_kv = outputs[2] if len(outputs) > 2 else None

        # attn_weights: [batch, num_heads, seq_q, seq_k]
        # Recompute value states (no RoPE on V in Llama).
        bsz, q_len, _ = hidden_states.size()
        v = self.v_proj(hidden_states) # [batch, seq_len, kv_heads*head_dim]                
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [batch, kv_heads, seq_len, head_dim]
        # Expand kv_heads to num_heads for GQA
        num_groups = self.num_heads // self.num_key_value_heads
        if num_groups > 1:
            v = v.repeat_interleave(num_groups, dim=1) # [batch, heads, seq_len, head_dim]
        last_weights = attn_weights[:, :, -1, :].unsqueeze(-1) # [batch, heads, seq_k, 1]
        self._last_attn_subvalues = last_weights * v # [batch, heads, seq_k, head_dim]
        self._last_attn_weights = attn_weights

        if not output_attentions:
            attn_weights = None
        
        if past_kv is not None:
            return attn_output, attn_weights, past_kv
        return attn_output, attn_weights

    LlamaAttention.forward = _patched_llama_attn

    # Patch 2: LlamaMLP.forward
    def _patched_llama_mlp(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x) # shape [batch, seq_len, ffn_dim]
        self._last_coefficient_scores = self.act_fn(gate) * up # shape [batch, seq_len, ffn_dim]
        return self.down_proj(self._last_coefficient_scores)

    LlamaMLP.forward = _patched_llama_mlp

    # Patch 3: LlamaDecoderLayer.forward
    for layer_i, layer in enumerate(model.model.layers):
        layer._layer_i = layer_i

    def _patched_llama_layer(self, hidden_states, attention_mask=None, position_ids=None,
                              past_key_value=None, output_attentions=False, use_cache=False,
                              **kwargs):
        layer_input = hidden_states # [batch, seq_len, hidden]

        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            **kwargs,
        )
        attn_out = attn_outputs[0]
        
        attn_subvalues = self.self_attn._last_attn_subvalues
        attn_weights_saved = self.self_attn._last_attn_weights
        
        hidden_states = residual + attn_out
        residual_output = hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_out = self.mlp(hidden_states)
        coeff_scores = self.mlp._last_coefficient_scores
        hidden_states = residual + ffn_out
        layer_output = hidden_states

        _states[self._layer_i] = dict(
            layer_input=layer_input[0],
            attn_output=attn_out[0],
            residual_output=residual_output[0],
            ffn_output=ffn_out[0],
            layer_output=layer_output[0],
            attn_subvalues=attn_subvalues[0],
            coeff_scores=coeff_scores[0],
            attn_weights=attn_weights_saved[0],
        )

        # Return (hidden_states, attn_weights, present) etc.
        return (hidden_states,) + attn_outputs[1:]

    LlamaDecoderLayer.forward = _patched_llama_layer


