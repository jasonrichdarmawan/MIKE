"""
Core concept of Logit Probability Increase (LPI) is to measure how much a specific neuron contributes to the model's confidence in predicting a target word.
"""

# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# %%

# 1. Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="auto")
model.eval()

# %%

# 2. Define the input and the target word we want to check (e.g., "Paris")
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
target_token_id = tokenizer.encode(" Paris")[0] # Notice the space before Paris

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    
# %%

# 3. Get the residual stream vector BEFORE the 5th layer (for the last token)
# Hidden states index 5 is the output of layer 4 (input to layer 5)
hl_minus_1 = outputs.hidden_states[5][0, -1, :]

# 4. Extract the FFN weights of Layer 5
# GPT-2 uses Conv1D, so weights are shaped differently than standard Linear layers
W_in = model.transformer.h[5].mlp.c_fc.weight     # Shape: (768, 3072) - The "Keys"
W_out = model.transformer.h[5].mlp.c_proj.weight  # Shape: (3072, 768) - The "Values"

neuron_importance_scores = []

for neuron_idx in range(W_in.shape[1]):
    # 5. Isolate a single "Neuron" (e.g., Neuron #42)
    subkey = W_in[:, neuron_idx]     # The Key to trigger this neuron
    subvalue = W_out[neuron_idx, :]  # The Value this neuron adds

    # Compute the coefficient score (Activation) for this neuron
    # Using Gelu activation (GPT-2's default)
    activation_score = F.gelu(torch.dot(hl_minus_1, subkey))

    # The actual vector this neuron contributes to the stream
    neuron_output_vector = activation_score * subvalue

    # 6. Calculate Log Probability Increase (Importance Score)
    unembedding_matrix = model.lm_head.weight # Maps 768-dim vector to vocabulary

    # Log prob without the neuron
    logits_base = torch.matmul(unembedding_matrix, hl_minus_1)
    log_prob_base = F.log_softmax(logits_base, dim=-1)[target_token_id]

    # Log prob WITH the neuron added to the stream
    logits_with_neuron = torch.matmul(unembedding_matrix, hl_minus_1 + neuron_output_vector)
    log_prob_with_neuron = F.log_softmax(logits_with_neuron, dim=-1)[target_token_id]

    # The Importance Score is the increase in log probability
    importance_score = log_prob_with_neuron - log_prob_base
    neuron_importance_scores.append(importance_score.item())

top_neuron_idx = torch.argmax(torch.tensor(neuron_importance_scores)).item()
importance_score = neuron_importance_scores[top_neuron_idx]
print(f"Neuron {top_neuron_idx} Importance Score for 'Paris': {importance_score:.4f}")

# top-300 neurons and values for "Paris"
top_k = 300
top_k_indices = torch.topk(torch.tensor(neuron_importance_scores), k=top_k).indices.tolist()
top_k_scores = {idx: neuron_importance_scores[idx] for idx in top_k_indices}
print(f"Top-{top_k} Importance Scores for 'Paris':\n{top_k_scores}")

# %%