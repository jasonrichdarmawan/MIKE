# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current focus (2026-05-04)

Running Revised_AlphaEdit (newly added editing method, hparams at `EasyEdit/hparams/Revised_AlphaEdit/llama3-8b.yaml`) on the 1015 `composition_incorrect` examples with `--neuron-only` to skip sublayer projection / knockout / PatchScopes. Goal: extend the substitute-vs-repair audit. AlphaEdit (now_correct n=211) tilts toward first-hop overlap (Jaccard 0.21 > 0.16); IFMET (n=21) tilts toward second-hop (0.18 > 0.07). Open question: does Revised_AlphaEdit produce a third pattern? Use `bash run_revised_alphaedit.sh` (or `WORKDIR=/workspace/CaKE bash run_revised_alphaedit.sh` on RunPod). Expected ~2.5h on A100.

## Environment

- Python 3.10, venv at `/lambda/nfs/kim/CaKE/venv` — use `/lambda/nfs/kim/CaKE/venv/bin/python`
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

# 4. Generate figures (use analyze_plots.py for aggregate plots,
#    analyze_plots_neuron.py for neuron-level plots)
python analysis/analyze_plots_neuron.py --model gpt2 --output-dir plots/gpt2 \
    --subsets correct:datasets/gpt2/composition_correct/neuron_attribution \
              incorrect:datasets/gpt2/composition_incorrect/neuron_attribution
```

For quick testing, add `--max-examples 20` to `neuron_attribution.py`. Checkpoints are saved every 50 examples by default (`--checkpoint-every`).

**Precompute covariance stats (required before AlphaEdit/AlphaEdit_Circuit, ~8h on A100):**
```bash
python -m EasyEdit.easyeditor.models.rome.layer_stats \
    --model_name meta-llama/Meta-Llama-3-8B \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
    --layer_tmp "model.layers.{}.mlp.down_proj" \
    --dataset wikipedia \
    --sample_size 100000 \
    --precision float32 \
    --stats_dir data/stats
```
Cached to `data/stats/`. AlphaEdit needs layers 4-8 only; AlphaEdit_Circuit needs all 32 since EAP selects layers per-request.

### Pipeline orchestrator (`analysis/run_analysis.py`)

Top-level driver: `python analysis/run_analysis.py --methods <method> ...` runs edit + eval + (optionally) analysis + restore for one or more methods. Methods: `preedit`, `alphaedit`, `ifmet`, `eap_alphaedit`, `revised_alphaedit`.

Skip flags: `--skip-analysis` (no analyses), `--skip-patchscopes`, `--skip-neuron-attribution`, `--neuron-only` (currently honoured by `run_revised_alphaedit` only — runs edit + eval + neuron attribution only, skipping sublayer projection / knockout / PatchScopes; ~2.5h vs ~3.5h).

`run_analysis.sh` runs all four post-edit methods (AlphaEdit → IFMET → EAP-AlphaEdit → Revised_AlphaEdit) sequentially. `run_revised_alphaedit.sh` runs only Revised_AlphaEdit. Both scripts honour `WORKDIR` and `VENV` env vars (defaults to lambda paths).

### HF data on `clke-results/CaKE`

- `kim/wikipedia_stats/` — covariance stats for Llama-3-8B layers 0–31 (~25 GB). Download into `data/stats/Meta-Llama-3-8B/wikipedia_stats/` to skip the ~8h precompute. No `null_space_project.pt` here — AlphaEdit recomputes it on first run, or grab from `jason/data/stats/Meta-Llama-3-8B/wikipedia_stats/null_space_project.pt`.
- `kim/datasets/meta-llama/Meta-Llama-3-8B/` — post-edit subsets from prior AlphaEdit and IFMET runs (`composition_incorrect_alphaedit/{now_correct,still_incorrect,...}`, `composition_incorrect_ifmet/...`). Includes `neuron_attribution/`, `entity_description/`, `attention_knockout/`, `sublayer_projection/`, `layers.csv`. Download with `hf download clke-results/CaKE --repo-type dataset --include "kim/datasets/*" --local-dir <tmp>` then move `kim/datasets/...` into the `datasets/` directory.

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

### Visualization (`analysis/analyze_plots_neuron.py`)

Reads CSVs from the neuron attribution step and produces publication-ready PDFs under `plots/{model}/`. Key plots: sublayer LPI profiles, head LPI heatmaps, chain layer profiles, cross-order Jaccard heatmaps (shared vmax across subsets), overlap position bars, overlap position×layer heatmaps.

Aggregate (non-neuron) plots — stages boxplot, PatchScopes heatmaps, min-decode-layer line — are produced by `analysis/analyze_plots.py`. Tabular summaries by `analysis/analyze_tables.py` (aggregate) and `analysis/analyze_tables_neuron.py` (neuron).

The `--heatmap-vmax` flag normalises Jaccard colour scale across all subsets. Cross-order heatmap axes: x = order -2→-1→0 (left to right), y = order -1→0 (bottom to top, `origin="lower"`).

### Two-hop prompt structure

Prompts follow: `[r2 tokens][r1 tokens][e1 tokens][last]`. Position labels: `e1` (entity 1), `r1` (inner relation), `r2` (outer relation), `last` (prediction position), `other`. The bridge entity e2 is never explicit in the prompt — it must be computed internally.

### EasyEdit submodule

Points to a fork at `kimthangk/EasyEdit` on the `kim-branch`. All Llama-3-8B hparams fixes plus the `Revised_AlphaEdit` method (`hparams/Revised_AlphaEdit/llama3-8b.yaml`, `easyeditor/models/revised_alphaedit/`) are committed there — `git submodule update --init --recursive` followed by `git -C EasyEdit checkout kim-branch` is sufficient, no manual patches needed.

Hparams available: `AlphaEdit/llama3-8b.yaml`, `AlphaEdit_Circuit/llama3-8b.yaml`, `IFMET/llama3-8b-shallow.yaml`, `IFMET/llama3-8b-deep.yaml`, `Revised_AlphaEdit/llama3-8b.yaml`.

### Model support

Both GPT-2 and Llama are supported. `patch_model.py` handles architecture differences. For Llama, GQA is handled via `kv_head_i = head_i // num_groups`. Non-contiguous tensor issue with Llama's o_proj is fixed with `.T.reshape()` instead of `.T.view()`.
