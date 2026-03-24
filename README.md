<h3 align="center"> CaKE: Circuit-aware Editing Enables Generalizable Knowledge  Learners </h3>

## Table of Contents
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- 🧐[Analyze](#analyze)
- 📚[Editing](#editing)
- 🌻[Acknowledgement](#acknowledgement)

---


## 🌟Overview
This work aims to improve the knowledge editing performance under the multi-hop reasoning settings.
## 🔧Installation

Install micromamba:
```
brew install micromamba
```

Note: You can use conda instead of micromamba. For our case, we have trouble in installing EasyEdit's requirements: `numpy==1.22.1` with conda.

Note: You can use conda-forge's mirror by editing `~/.condarc`, removing `- conda-forge` and `- defaults` in the `channels` key. For our case, we have trouble in installing `pytorch==2.0.1`.
```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
show_channel_urls: true
```

Note: You can use PyPI's mirror. For our case, we have trouble in installing `zhipuai`:
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Build the environement:
```
micromamba create -n multihop python=3.10

micromamba activate multihop

micromamba install datasets==3.3.1 einops==0.4.0 gpustat==1.1 hydra-core==1.1.1 higher==0.2.1 importlib-metadata==6.3.0 matplotlib==3.5.1 nltk==3.6.5 numpy==1.22.1 omegaconf==2.1.1 pandas==1.4.0 pyyaml==6.0 scikit-learn==1.0.2 scipy==1.7.3 sentence-transformers==3.2.1 tokenizers==0.19.1
```

```
micromamba install -c nvidia pytorch==2.1.0
```

```
micromamba install tqdm==4.62.3 transformers==4.44.2 openai==2.24.0 peft==0.7.1 timm==0.9.7 iopath==0.1.10 sentencepiece seaborn

pip install opencv-python==4.8.0.76 fairscale==0.4.13

micromamba install beartype jaxtyping

pip install baukit zhipuai sniffio
```

## 📚Analyze
Download the wikidata for analysis from [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/blob/main/datasets/two_hop.csv) into `datasets`.

### Pipeline overview

```
1. run_preedit_analysis.py      — evaluate base model, create subsets, run all analyses
2. neuron_attribution.py        — neuron-level LPI and backward attribution (GPU recommended)
3. run_alphaedit_analysis.py    — apply AlphaEdit, evaluate, split subsets, run all analyses
   run_ifmet_analysis.py        — same for IFMET
4. neuron_attribution.py        — same, with --checkpoint for post-edit weights
5. analyze_tables.py            — print summary tables (mechanism stages)
   analyze_tables_neuron.py     — print neuron attribution tables
   analyze_plots.py           — generate mechanism plots
   analyze_plots_neuron.py    — generate neuron attribution plots
```

Each run script skips steps whose output files already exist, so re-runs are safe.
Classification results are cached automatically as `*_classified.csv` on first run.

---

### Pre-edit analysis

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B

python analysis/run_preedit_analysis.py \
    --model $MODEL \
    --input datasets/two_hop.csv \
    --data-dir $DATA_DIR
```

Creates three subsets under `$DATA_DIR/composition_{correct,incorrect,inconsistent}/two_hop.csv`.
Analyses (methods 1–4) are run only on `correct` and `incorrect`; `inconsistent` is saved as CSV only.

Each analysed subset gets:
- `sublayer_projection/{attention,mlp}_projections.csv`
- `attention_knockout/last_e1_attention_knockout_k3_composition.csv`
- `entity_description/{e1,last}_entity_description_s0.csv`

Add `--skip-patchscopes` to skip PatchScopes (method 4, slowest).

---

### Neuron attribution

Requires GPU (~2 min for 900 examples). Uses `analysis/patch_model.py` to monkey-patch GPT-2/Llama forward passes without modifying model files.

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B

# Pre-edit
python analysis/neuron_attribution.py \
    --model-name $MODEL --data-dir $DATA_DIR \
    --prompts two-hop first-hop second-hop \
    --subsets correct incorrect

# Post-edit (load edited weights via checkpoint)
python analysis/neuron_attribution.py \
    --model-name $MODEL --data-dir $DATA_DIR/composition_incorrect_alphaedit \
    --checkpoint $DATA_DIR/composition_incorrect_alphaedit/editing_checkpoint.pt \
    --prompts two-hop first-hop second-hop \
    --subsets now_correct still_incorrect

# Quick test
python analysis/neuron_attribution.py \
    --model-name $MODEL --data-dir $DATA_DIR \
    --prompts two-hop --subsets correct --max-examples 20
```

Outputs per `(subset, prompt_type)` under `$DATA_DIR/{subset}/neuron_attribution/{prompt_type}/`:
- `sublayer_lpi.csv`, `head_lpi.csv` — per-example LPI
- `ffn_query_layers.csv`, `attn_query_layers.csv` — residual stream decomposition
- `chain_profile.csv`, `chain_mean_layer.csv` — aggregated chain attribution
- `chain_position_dist.csv` — marginal position distribution

Under `$DATA_DIR/{subset}/neuron_attribution/`:
- `jaccard.csv`, `overlap_position.csv`, `overlap_position_by_layer.csv`

---

### Post-edit analysis

#### YAML configuration

Both AlphaEdit and IFMET require hparams YAML files. Llama-3-8B files are already included:

| Method | YAML path |
|--------|-----------|
| AlphaEdit | `EasyEdit/hparams/AlphaEdit/llama3-8b.yaml` |
| IFMET phase 1 (shallow layers 4–8) | `EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml` |
| IFMET phase 2 (deep layers 13–17) | `EasyEdit/hparams/IFMET/llama3-8b-deep.yaml` |

Key fields to check before running:

- **`device`** — GPU index (default `0`). Must match `--device` CLI arg passed to the run script.
- **`stats_dir`** — Directory where Fisher/covariance statistics are cached (default `./data/stats`). Computed once on first run (can take tens of minutes); reused on subsequent runs.
- **`P_loc`** *(AlphaEdit only)* — Path to the null-space projection matrix (default `./null_space_project.pt`). Auto-computed and saved on first run if the file is absent. Can also be downloaded from the link in `EasyEdit/easyeditor/models/alphaedit/README.md`.

For a **new model**, copy the closest existing YAML, update `model_name`, `layers`, `v_loss_layer` (= last layer index), and the `*_module_tmp` / `ln_f_module` / `lm_head_module` fields to match the model's architecture.

#### Running

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B

# AlphaEdit
python analysis/run_alphaedit_analysis.py \
    --model $MODEL \
    --hparams EasyEdit/hparams/AlphaEdit/llama3-8b.yaml \
    --input $DATA_DIR/composition_incorrect/two_hop.csv \
    --output $DATA_DIR/composition_incorrect_alphaedit

# IFMET
python analysis/run_ifmet_analysis.py \
    --model $MODEL \
    --hparams-shallow EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml \
    --hparams-deep    EasyEdit/hparams/IFMET/llama3-8b-deep.yaml \
    --input $DATA_DIR/composition_incorrect/two_hop.csv \
    --output $DATA_DIR/composition_incorrect_ifmet
```

Creates three subsets under `$OUTPUT/{now_correct,still_incorrect,still_inconsistent}/two_hop.csv`.
Analyses are run only on `now_correct` and `still_incorrect`; `still_inconsistent` is saved as CSV only.

---

### Tables and plots

All `analyze_*` scripts accept `--subsets name:path ...` and work for any combination of pre- and post-edit subsets.

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B
PLOTS=plots/meta-llama/Meta-Llama-3-8B

# Summary tables (mechanism)
python analysis/analyze_tables.py --subsets \
    correct:$DATA_DIR/composition_correct \
    incorrect:$DATA_DIR/composition_incorrect

# Summary tables (neuron attribution)
python analysis/analyze_tables_neuron.py --subsets \
    correct:$DATA_DIR/correct/neuron_attribution \
    incorrect:$DATA_DIR/incorrect/neuron_attribution

# Mechanism plots
python analysis/analyze_plots.py \
    --model $MODEL --output-dir $PLOTS \
    --subsets \
        correct:$DATA_DIR/composition_correct \
        incorrect:$DATA_DIR/composition_incorrect

# Neuron attribution plots
python analysis/analyze_plots_neuron.py \
    --model $MODEL --output-dir $PLOTS \
    --subsets \
        correct:$DATA_DIR/correct/neuron_attribution \
        incorrect:$DATA_DIR/incorrect/neuron_attribution

# Pre- vs post-edit comparison
python analysis/analyze_tables.py --subsets \
    pre-edit:$DATA_DIR/composition_incorrect \
    now_correct:$DATA_DIR/composition_incorrect_alphaedit/now_correct \
    still_incorrect:$DATA_DIR/composition_incorrect_alphaedit/still_incorrect
```

---

### Script reference

**Orchestrators** — load the model once and run the full analysis pipeline end-to-end:

| Script | Purpose |
|--------|---------|
| `run_preedit_analysis.py` | Evaluate base model, create correct/incorrect/inconsistent subsets, run all 4 analyses on correct and incorrect only |
| `run_alphaedit_analysis.py` | Apply AlphaEdit, evaluate edited model, split into now_correct/still_incorrect/still_inconsistent, run all 4 analyses on now_correct and still_incorrect only |
| `run_ifmet_analysis.py` | Same as above for IFMET (two-phase MEMIT); analyses on now_correct and still_incorrect only |

**Analysis methods** — called by the orchestrators; can also be run standalone with `--checkpoint` for a pre-saved edited model:

| Script | Purpose |
|--------|---------|
| `generate_entity_description.py` | PatchScopes: patch hidden state at source position, decode entity description |
| `project_sublayer.py` | Sublayer projection: project attention/MLP output into vocabulary per layer |
| `knockout_attention.py` | Attention knockout: zero out attention from source to target, measure answer change |
| `evaluate_dataset.py` | Evaluate model on two_hop.csv (first-hop, second-hop, composition, shortcuts) |
| `neuron_attribution.py` | Neuron-level LPI and backward attribution chain (GPU recommended) |

**Analysis output scripts** — read saved CSVs and produce tables or plots:

| Script | Purpose |
|--------|---------|
| `analyze_tables.py` | Build `layers.csv` from source CSVs (cached), print mechanism stage tables |
| `analyze_tables_neuron.py` | Print neuron attribution tables (LPI, chain, Jaccard, position) |
| `analyze_plots.py` | Generate mechanism plots (stage boxplots, PatchScopes heatmaps) |
| `analyze_plots_neuron.py` | Generate neuron attribution plots (LPI profiles, chain heatmaps, overlap bars) |

**Internal modules** — imported by the scripts above, not run directly:

| Script | Purpose |
|--------|---------|
| `patch_model.py` | Monkey-patches GPT-2/Llama forward passes to capture internal states |
| `patched_generation.py` | Low-level generation with attention knockout and patching hooks; KV-cache variant gives ~32× speedup for PatchScopes |
| `classify_results.py` | Classify raw CSV outputs; results cached as `*_classified.csv` |
| `utils.py` | Model loading, answer checking, tokenizer helpers |

**Standalone utilities:**

| Script | Purpose |
|--------|---------|
| `llm_translate.py` | Translate MQuAKE dataset entries via LLM API |

**Not part of the current pipeline:**

| Script | Status |
|--------|--------|
| `patch_activations.py` | Superseded by `generate_entity_description.py` |
| `create_subset.py` | Superseded — logic inlined into `run_preedit_analysis.py` |
| `analyze_subset.py` | Depends on `patch_activations.py`; not used in current pipeline |
| `neuron_importance.py` | Early prototype; superseded by `neuron_attribution.py` |

---

### MQuAKE multilingual benchmark

- Copy `datasets/MQuAKE-CF-3k-v2-cake.json` to `datasets/MQuAKE-CF-3k-v2-clke.json`
- Run `analysis/llm_translate.py` to translate:
  ```bash
  python analysis/llm_translate.py \
      --dataset_path datasets/MQuAKE-CF-3k-v2-clke.json \
      --model gpt-oss-120b \
      --endpoint <YOUR_ENDPOINT> --api_key <YOUR_API_KEY> \
      --case_id 1 1 --language zh
  ```

[1] E. Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
[2] Y. Yao et al., "CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners," in EMNLP 2025
[3] Z. Yu et al., "Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models," in EMNLP 2025
[4] Z. Yu et al., "Neuron-Level Knowledge Attribution in Large Language Models," in EMNLP 2024

## 🧐Editing
Just run the following commond:
```
run.sh
```

## 🌻Acknowledgement

We thank for the project of [EasyEdit](https://github.com/zjunlp/EasyEdit), [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate).
The code in this work is built on top of these projects' codes.