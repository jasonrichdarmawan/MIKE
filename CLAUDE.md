# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.10, venv at `/lambda/nfs/Llama-3-8B-Analysis/venv` — use `/lambda/nfs/Llama-3-8B-Analysis/venv/bin/python`
- GPU required for practical runtimes. Llama-3-8B on A100 40GB rough timings:
  - Edit + full analysis (AlphaEdit): ~3.5h (editing ~45m, eval ~15m, sublayer/knockout ~1h, PatchScopes ~90m)
  - Edit + full analysis (IFMET): ~4.5h
  - Neuron attribution (900 ex × 3 prompts × 3 subsets): ~1.5h per method
- EasyEdit is a git submodule (`EasyEdit/`) — run `git submodule update --init` after cloning, then apply the local patches described below
- Key packages: `transformers==4.44.2`, `pytorch==2.1.0`, `peft==0.7.1`, `pandas==1.4.0`
- Model (`meta-llama/Meta-Llama-3-8B`) is gated on HuggingFace — run `huggingface-cli login` before first use

## Running

**Knowledge editing:**
```bash
bash run.sh
# or directly:
python test_cake.py --editing_method CAKE --datatype MQuAKE-CF-3k --model_type qwen2.5-7b
```

**Analysis pipeline (in order):**
```bash
# 1. Evaluate base model on dataset
python analysis/evaluate_dataset.py gpt2 --input_path datasets/two_hop.csv --output_path datasets/gpt2/two_hop.csv

# 2. Create correct/incorrect/inconsistent subsets
python analysis/create_subset.py

# 3. Neuron attribution (GPU recommended)
python analysis/neuron_attribution.py --model-name gpt2 --data-dir datasets/gpt2 \
    --prompts two-hop first-hop second-hop --subsets correct incorrect inconsistent

# 4. Generate figures
python analysis/analyze_neuron_attribution.py datasets plots -m gpt2
```

For quick testing, add `--max-examples 20` to `neuron_attribution.py`. Checkpoints are saved every 50 examples by default (`--checkpoint-every`).

## Architecture

### Two independent tracks

**Editing track** (`test_cake.py` + `edit_utils.py`):
- Implements CaKE (LoRA on 7 projection modules: q/k/v/o_proj, up/down/gate_proj), ROME, MEMIT, WISE, LoRA, IFMET, MeLLo
- Input: `datasets/{datatype}-cake.json`; Output: `output/{method}_{model}_{datatype}_metrics.json`
- EasyEdit submodule provides base implementations; `edit_utils.py` wraps them with CaKE-specific logic

**Analysis track** (`analysis/`):
- Dataset: `datasets/two_hop.csv` from HoppingTooLate project
- Three subsets: `correct` (both hops + composition correct), `incorrect` (hops correct, composition fails), `inconsistent`
- Output lives under `datasets/{model}/neuron_attribution/{subset}/` and `plots/{model}/`

### Model patching (`analysis/patch_model.py`)

Central to the analysis track. Monkey-patches GPT-2 and Llama forward passes to capture internal states (`layer_input`, `attn_output`, `ffn_output`, `coeff_scores`, `attn_subvalues`, `residual_output`) without modifying model files. Called via `patch_model.patch(model)` before running any analysis. `patch_model.extract_states(layer_num)` retrieves states after each forward pass.

### Neuron attribution (`analysis/neuron_attribution.py`)

The core analysis script. For each example and prompt type, runs one forward pass and computes:
- **LPI (Log Probability Increase)**: per-sublayer and per-head causal contribution to the predicted token
- **Top-K neurons**: global top-K FFN and attention value neurons ranked by LPI
- **Backward attribution chain**: order 0 (value neurons) → order -1 (which neurons activated them) → order -2; chain ntype labels are `"ffn"`, `"attn"` at order 0; `"attn→ffn"`, `"ffn→attn"` at order -1; `"ffn→attn"` only at order -2
- **Jaccard similarity**: on-the-fly cross-prompt-type overlap per layer, saved to `jaccard.csv`
- **Position tracking**: where in the sequence (e1/r1/r2/last/other) neurons fire, including joint position×layer for overlapping neurons

`neuron_class` groups same-class neurons: `"ffn"` covers `"ffn"` and `"attn→ffn"`; `"attn"` covers `"attn"` and `"ffn→attn"`. This is used in Jaccard filtering to allow cross-ntype same-class comparisons.

Order convention: 0 = value neurons, -1 = first backward step, -2 = second backward step (negative = backward from prediction).

### Output files per `(subset, prompt_type)`

Under `datasets/{model}/neuron_attribution/{subset}/{prompt_type}/`:
- `sublayer_lpi.csv`, `head_lpi.csv` — per-example LPI
- `ffn_query_layers.csv`, `attn_query_layers.csv` — residual stream decomposition
- `chain_profile.csv`, `chain_mean_layer.csv` — aggregated chain attribution
- `chain_position_dist.csv` — marginal position distribution of neurons

Under `datasets/{model}/neuron_attribution/{subset}/`:
- `jaccard.csv` — mean Jaccard per `(pair, neuron_class, ntype_a, order_a, ntype_b, order_b, layer)`
- `overlap_position.csv` — position of overlapping neurons, collapsed over layers
- `overlap_position_by_layer.csv` — joint position × layer for overlapping neurons

### Visualization (`analysis/analyze_neuron_attribution.py`)

Reads CSVs from the neuron attribution step and produces publication-ready PDFs under `plots/{model}/`. Key plots: sublayer LPI profiles, head LPI heatmaps, chain layer profiles, cross-order Jaccard heatmaps (shared vmax across subsets), overlap position bars, overlap position×layer heatmaps.

The `--heatmap-vmax` flag normalises Jaccard colour scale across all subsets. Cross-order heatmap axes: x = order -2→-1→0 (left to right), y = order -1→0 (bottom to top, `origin="lower"`).

### Two-hop prompt structure

Prompts follow: `[r2 tokens][r1 tokens][e1 tokens][last]`. Position labels: `e1` (entity 1), `r1` (inner relation), `r2` (outer relation), `last` (prediction position), `other`. The bridge entity e2 is never explicit in the prompt — it must be computed internally.

### Model support

### Local changes to EasyEdit submodule

The EasyEdit submodule (`278ad57`) points to a fork at `jasonrichdarmawan/EasyEdit`. The following hparams files have local modifications for Llama-3-8B that should be committed to the fork:

1. **`hparams/AlphaEdit/llama3-8b.yaml`**
   - Fixed `model_name` from `./hugging_cache/llama-3-8b-instruct` to `meta-llama/Meta-Llama-3-8B`
   - Added `mom2_batch_tokens: None` (new required field in `AlphaEditHyperParams`)

2. **`hparams/AlphaEdit_Circuit/llama3-8b.yaml`**
   - Added `num_hidden_layers: 32` (required by `AlphaEditCircuitHyperParams`)
   - Added `hook_q_input`, `hook_k_input`, `hook_v_input`, `hook_mlp_in`, `hook_resid_post` fields (required by hparams dataclass)

3. **`hparams/IFMET/`** (new directory)
   - `llama3-8b-shallow.yaml` — MEMIT hparams for standard r1/r2 edits (layers 4-8)
   - `llama3-8b-deep.yaml` — MEMIT hparams for indirect composition edits (layers 20-24)


### Model support

Both GPT-2 and Llama are supported. `patch_model.py` handles architecture differences. For Llama, GQA is handled via `kv_head_i = head_i // num_groups`. Non-contiguous tensor issue with Llama's o_proj is fixed with `.T.reshape()` instead of `.T.view()`.
