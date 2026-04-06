# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

CaKE (Circuit-aware Knowledge Editing) research codebase. The primary focus is the **analysis pipeline** in `analysis/`, which investigates mechanistic differences between correct/incorrect multi-hop compositions using sublayer projection, attention knockout, PatchScopes, and neuron attribution — built on top of the HoppingTooLate two-hop dataset (`datasets/two_hop.csv`).

There is also an editing pipeline (`test_cake.py`, `edit_utils.py`) for running knowledge editing methods (CaKE, MEMIT, IFMET, LoRA, WISE, ROME, MeLLo) on MQuAKE datasets. Both pipelines use `EasyEdit/` as a git submodule.

## Environment setup

```bash
micromamba create -n multihop python=3.10
micromamba activate multihop
micromamba install datasets==3.3.1 einops==0.4.0 gpustat==1.1 hydra-core==1.1.1 higher==0.2.1 importlib-metadata==6.3.0 matplotlib==3.5.1 nltk==3.6.5 numpy==1.22.1 omegaconf==2.1.1 pandas==1.4.0 pyyaml==6.0 scikit-learn==1.0.2 scipy==1.7.3 sentence-transformers==3.2.1 tokenizers==0.19.1
micromamba install -c nvidia pytorch==2.1.0
micromamba install tqdm==4.62.3 transformers==4.44.2 openai==2.24.0 peft==0.7.1 timm==0.9.7 iopath==0.1.10 sentencepiece seaborn
pip install opencv-python==4.8.0.76 fairscale==0.4.13
micromamba install beartype jaxtyping
pip install baukit zhipuai sniffio
```

Set `HF_TOKEN` env var for gated models (e.g., Llama).

## Common commands

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B

# Pre-edit: evaluate base model, create subsets, run mechanism analyses
python analysis/run_analysis.py --model $MODEL --data-dir $DATA_DIR

# Post-edit (AlphaEdit)
python analysis/run_analysis.py \
    --model $MODEL --data-dir $DATA_DIR --methods alphaedit \
    --hparams-alphaedit EasyEdit/hparams/AlphaEdit/llama3-8b.yaml

# Post-edit (IFMET)
python analysis/run_analysis.py \
    --model $MODEL --data-dir $DATA_DIR --methods ifmet \
    --hparams-ifmet-shallow EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml \
    --hparams-ifmet-deep    EasyEdit/hparams/IFMET/llama3-8b-deep.yaml

# Full pipeline (edit → all analyses → HF push)
bash run_overnight.sh

# Neuron attribution is now integrated into run_analysis.py.
# To skip it: --skip-neuron-attribution
# To run only specific prompt types: --na-prompts two-hop
# Standalone (if needed):
python analysis/neuron_attribution.py \
    --model-name $MODEL --data-dir $DATA_DIR \
    --prompts two-hop --subsets correct --max-examples 20

# Tables and plots (pass any named subset:path pairs)
python analysis/analyze_tables.py --subsets \
    correct:$DATA_DIR/composition_correct \
    incorrect:$DATA_DIR/composition_incorrect

python analysis/analyze_tables_neuron.py --subsets \
    correct:$DATA_DIR/composition_correct/neuron_attribution \
    incorrect:$DATA_DIR/composition_incorrect/neuron_attribution

python analysis/analyze_plots.py --model $MODEL --output-dir plots/... \
    --subsets correct:$DATA_DIR/composition_correct incorrect:$DATA_DIR/composition_incorrect

python analysis/analyze_plots_neuron.py --model $MODEL --output-dir plots/... \
    --subsets correct:$DATA_DIR/composition_correct/neuron_attribution \
            incorrect:$DATA_DIR/composition_incorrect/neuron_attribution
```

`run_analysis.py` is idempotent — skips steps whose output files already exist. Delete output files to force a re-run.

## Analysis pipeline architecture

### Data flow

```
datasets/two_hop.csv
  └─ run_analysis.py (pre-edit)
       ├─ evaluate_dataset.py → DATA_DIR/two_hop.csv  (adds *_correct columns)
       ├─ create_subsets()    → composition_{correct,incorrect,inconsistent}/two_hop.csv
       └─ for each subset in {correct, incorrect}:
            ├─ project_sublayer.py       → sublayer_projection/sublayer_projection.csv
            ├─ knockout_attention.py     → attention_knockout/attention_knockout_k*.csv
            ├─ generate_entity_description.py → entity_description/entity_description_s*.csv
            └─ neuron_attribution.py     → neuron_attribution/{prompt_type}/*.csv

  └─ run_analysis.py (post-edit, per method)
       ├─ apply editing algorithm (modifies model weights in-place, restores afterward)
       ├─ evaluate_dataset.py → OUT_DIR/two_hop.csv
       ├─ split_subsets()     → {now_correct,still_incorrect,still_inconsistent,broken_hops}/two_hop.csv
       └─ for each subset in {now_correct, still_incorrect}:
            └─ same four analyses as pre-edit

  └─ neuron_attribution.py  (also runnable standalone, GPU required)
       └─ {subset}/neuron_attribution/{prompt_type}/
            ├─ sublayer_lpi.csv, head_lpi.csv
            ├─ ffn_query_layers.csv, attn_query_layers.csv
            ├─ chain_profile.csv, chain_mean_layer.csv
            └─ chain_position_dist.csv
          {subset}/neuron_attribution/jaccard.csv  (cross-prompt overlap)

  └─ analyze_tables.py / analyze_plots.py
       ├─ loads CSVs from subset dirs
       ├─ calls classify_results.py (caches as *_classified.csv)
       └─ prints tables / saves plots
```

### Script roles

**Orchestrators:**
- `run_analysis.py`: Loads the model once per method, runs eval → subset creation → all four mechanism analyses. Accepts `--skip-analysis` to only edit+eval, `--skip-patchscopes` to skip entity description, `--skip-neuron-attribution` to skip neuron attribution, `--na-prompts` to select prompt types. Use `--max-examples N` for quick tests.

**Mechanism analysis scripts** (called by orchestrator; also runnable standalone):
- `evaluate_dataset.py`: Adds `first_hop_correct`, `second_hop_correct`, `composition_correct`, `entity_shortcut_correct`, `relation_shortcut_correct` columns via batched generation. Supports `--checkpoint` to load edited weights.
- `project_sublayer.py`: Projects per-layer attention and MLP outputs into vocabulary space; finds which layer first promotes e2/e3 tokens.
- `knockout_attention.py`: Zeros out attention from source positions to target token positions; measures how answer changes across k values.
- `generate_entity_description.py`: PatchScopes — patches hidden state at entity position from a source prompt into a target context (`TARGET_PROMPT` in `run_analysis.py`), then decodes what entity the model sees at that layer.
- `neuron_attribution.py`: LPI (Layer-wise Prediction Impact) and chained backward attribution at neuron level. Computes FFN/attention neuron importance and cross-prompt Jaccard overlap. Requires GPU. Exposes `setup_attribution`, `process_examples`, `aggregate_and_save` for use by `run_analysis.py`; also runnable standalone.

**Output scripts:**
- `analyze_tables.py`: Builds `layers.csv` per subset (merged from sublayer projection, attention knockout, entity description CSVs via `classify_results.py`), then prints mechanism stage summary tables.
- `analyze_tables_neuron.py`: Prints LPI, chain attribution, Jaccard, and position distribution tables from neuron attribution CSVs.
- `analyze_plots.py` / `analyze_plots_neuron.py`: Generate plots from the same CSVs.

**Infrastructure:**
- `patch_model.py`: Monkey-patches GPT-2/Llama forward passes in-process (no file changes) to capture `layer_input`, `attn_output`, `ffn_output`, `attn_subvalues`, `attn_weights` per layer via `_states` dict. Call `patch_model.patch(model)` then run a forward pass, then `patch_model.extract_states(layer_num)`.
- `patched_generation.py`: Low-level generation with knockout/patching hooks; KV-cache variant gives ~32x speedup for PatchScopes batch runs.
- `classify_results.py`: Classifies raw generation CSVs into entity labels (e1/e2/e3); results cached as `*_classified.csv`. Delete to force reclassification.
- `utils.py`: `load_model`, `load_tokenizer`, model-type dispatch for layer name accessors (`get_layer_names`, `get_mlp_layers_names`, etc.), `generate_and_test_answers`, `check_answer_in_pred`. Supports GPT-2, Llama-2/3, Gemma, Qwen2.

### Supported models

`utils.py` and `patch_model.py` dispatch on `type(model)`. Currently supported: `GPT2LMHeadModel`, `LlamaForCausalLM`, `GemmaForCausalLM`, `Qwen2ForCausalLM`. Adding a new model requires updating both files' dispatch blocks plus `patch_model._patch_*` internals.

### Classification caching

Any `*_classified.csv` is a cached output of `classify_results.py`. Delete it to force reclassification. The `layers.csv` built by `analyze_tables.py` is similarly cached per subset directory.

### Subset definitions

**Pre-edit:**
| Subset | Definition |
|--------|------------|
| `correct` | composition + both hops correct, no shortcuts (n=100/type) |
| `incorrect` | both hops correct, composition fails in all contexts (n=50/type) |
| `inconsistent` | both hops correct, composition fails; bridge entity e2 is solvable in some other r2 context (n=30/type) |

**Post-edit:**
| Subset | Definition |
|--------|------------|
| `now_correct` | Both hops + composition pass after editing |
| `still_incorrect` | Both hops pass, composition still fails |
| `still_inconsistent` | Both hops pass, composition fails; bridge entity decodable in now_correct |
| `broken_hops` | One or both single-hop retrievals broke after editing |

Mechanism analyses (sublayer, knockout, patchscopes, neuron attribution) run only on `correct`+`incorrect` (pre-edit) and `now_correct`+`still_incorrect` (post-edit).
