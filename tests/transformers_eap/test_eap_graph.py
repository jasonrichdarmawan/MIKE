import torch
import pytest
from transformers import GPT2Config, AutoModelForCausalLM

from transformers_eap.eap_graph import EAPGraph

def test_gpt2_mlp_backprop():
    print("Testing GPT-2 EAP Graph initialization and hooks...")
    config = GPT2Config(
        n_layer=2, 
        n_head=4, 
        n_embd=32, 
        vocab_size=100
    )
    model = AutoModelForCausalLM.from_config(config)
    
    eap = EAPGraph(model, graph_type="gpt2")
    
    # 1. Check Component Getter
    comps, _ = eap._get_layer_components(0)
    assert "ln_2" in comps, "GPT-2 should have ln_2 in components"
    assert comps["ln_2"] is not None
    print("Component getter passed.")
    
    # 2. Check Forward Hooks Registration
    eap.register_forward_hooks()
    
    # Verify we have hooks
    input_ids = torch.randint(0, 100, (1, 10))
    eap.model(input_ids)
    
    resid_mid_keys = [k for k in eap.activations.keys() if "resid_mid" in k]
    assert len(resid_mid_keys) > 0, "Should have captured resid_mid activations for MLP attribution"
    
    # 3. Check Backward Pass
    eap.register_backward_hooks()
    
    logits = eap.model(input_ids).logits
    loss = logits.sum()
    loss.backward()
    
    mlp_in_grads = [k for k in eap.gradients.keys() if "mlp_in" in k]
    assert len(mlp_in_grads) > 0, "Should have captured mlp_in gradients"
    print("Test GPT-2 MLP Backprop passed.")

def test_llama_mlp_backprop():
    # Only test if LlamaConfig is available (transformers library update)
    try:
        from transformers import LlamaConfig
    except ImportError:
        pytest.skip("LlamaConfig not available in this transformers version")
        return

    print("\nTesting Llama EAP Graph initialization and hooks...")
    config = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=32,
        vocab_size=100,
        intermediate_size=64
    )
    model = AutoModelForCausalLM.from_config(config)
    
    eap = EAPGraph(model, graph_type="llama_like")
    
    # 1. Check Component Getter
    comps, _ = eap._get_layer_components(0)
    # Llama maps post_attention_layernorm to ln_2
    assert "ln_2" in comps, "Llama-like should return ln_2 component (mapped from post_attn_ln)"
    
    # 2. Check Forward Hooks Registration
    eap.register_forward_hooks()
    
    input_ids = torch.randint(0, 100, (1, 10))
    eap.model(input_ids)
    
    # In Llama, ln_2 is post_attention_layernorm
    resid_mid_keys = [k for k in eap.activations.keys() if "resid_mid" in k]
    assert len(resid_mid_keys) > 0, "Should have captured resid_mid (post-attn LN input)"
    
    # 3. Check Backward Pass
    eap.register_backward_hooks()
    
    logits = eap.model(input_ids).logits
    loss = logits.sum()
    loss.backward()
    
    mlp_in_grads = [k for k in eap.gradients.keys() if "mlp_in" in k]
    assert len(mlp_in_grads) > 0
    print("Test Llama MLP Backprop passed.")

if __name__ == "__main__":
    test_gpt2_mlp_backprop()
    test_llama_mlp_backprop()
