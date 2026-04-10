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

Install `EasyEdit` repository as submodule
```
git submodule update --init --recursive
```

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

pip install SPARQLWrapper==2.0.0

micromamba install datasets==3.3.1 einops==0.4.0 gpustat==1.1 hydra-core==1.1.1 higher==0.2.1 importlib-metadata==6.3.0 matplotlib==3.5.1 nltk==3.6.5 omegaconf==2.1.1 pandas==1.4.0 pyyaml==6.0 scikit-learn==1.0.2 tokenizers==0.19.1

pip install "numpy>=1.24,<2" scipy sentence-transformers==3.2.1

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install transformers==4.57.6

micromamba install tqdm==4.62.3 openai==2.24.0 peft==0.7.1 timm==0.9.7 iopath==0.1.10 sentencepiece seaborn

pip install opencv-python==4.8.0.76 fairscale==0.4.13

micromamba install beartype jaxtyping pytest

pip install baukit zhipuai sniffio
```

## 📚Analyze
Download the wikidata for analysis from [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/blob/main/datasets/two_hop.csv) into `datasets`.

### Pipeline overview

```
run_overnight.sh
  ├── run_analysis.py --methods preedit    — evaluate, create subsets, all 4 analyses
  ├── run_analysis.py --methods ifmet      — edit, evaluate, split subsets, all 4 analyses
  └── run_analysis.py --methods alphaedit  — same as IFMET

analyze_tables.py / analyze_plots.py      — tables and plots from saved CSVs
```

The four analyses are: sublayer projection, attention knockout, PatchScopes, and neuron attribution.
Neuron attribution is now integrated into `run_analysis.py` (previously required separate runs).

`run_analysis.py` skips steps whose output files already exist, so re-runs are safe.
Classification results are cached automatically as `*_classified.csv` on first run.

---

### Running the full pipeline

```bash
bash run_overnight.sh   # full end-to-end (edit -> neuron attribution -> push to HF)
```

Or step by step:

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B

# Pre-edit: evaluate base model, create correct/incorrect/inconsistent subsets, run analyses
python analysis/run_analysis.py \
    --model $MODEL --data-dir $DATA_DIR

# Post-edit: AlphaEdit -- edit, evaluate, split subsets, run analyses
python analysis/run_analysis.py \
    --model $MODEL --data-dir $DATA_DIR --methods alphaedit \
    --hparams-alphaedit EasyEdit/hparams/AlphaEdit/llama3-8b.yaml

# Post-edit: IFMET
python analysis/run_analysis.py \
    --model $MODEL --data-dir $DATA_DIR --methods ifmet \
    --hparams-ifmet-shallow EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml \
    --hparams-ifmet-deep    EasyEdit/hparams/IFMET/llama3-8b-deep.yaml
```

Pre-edit creates three subsets under `$DATA_DIR/composition_{correct,incorrect,inconsistent}/two_hop.csv`.
Post-edit creates four subsets under `$OUT/{now_correct,still_incorrect,still_inconsistent,broken_hops}/two_hop.csv`:

| Subset | Definition |
|--------|------------|
| `now_correct` | Both single hops and composition pass after editing |
| `still_incorrect` | Both single hops pass, composition still fails |
| `still_inconsistent` | Both single hops pass, composition fails due to inconsistent second-hop |
| `broken_hops` | One or both single-hop retrievals broke after editing |

Mechanism analyses (sublayer projection, attention knockout, patchscopes, neuron attribution) run on `now_correct` and `still_incorrect` only; `still_inconsistent` and `broken_hops` are saved as CSVs only.

Use `--skip-neuron-attribution` to skip neuron attribution, `--skip-patchscopes` to skip PatchScopes, or `--skip-analysis` to skip all analyses. Use `--na-prompts two-hop` to run neuron attribution on a single prompt type.

#### YAML configuration

Both AlphaEdit and IFMET require hparams YAML files. Llama-3-8B files are already included:

| Method | YAML path |
|--------|-----------|
| AlphaEdit | `EasyEdit/hparams/AlphaEdit/llama3-8b.yaml` |
| IFMET phase 1 (shallow layers 4-8) | `EasyEdit/hparams/IFMET/llama3-8b-shallow.yaml` |
| IFMET phase 2 (deep layers 13-17) | `EasyEdit/hparams/IFMET/llama3-8b-deep.yaml` |

Key fields to check before running:

- **`device`** -- GPU index (default `0`).
- **`stats_dir`** -- Directory where Fisher/covariance statistics are cached (default `./data/stats`). Computed once on first run; reused on subsequent runs.
- **`P_loc`** *(AlphaEdit only)* -- Path to the null-space projection matrix (default `./null_space_project.pt`). Auto-computed on first run if absent.

---

### Neuron attribution

Neuron attribution is now integrated into `run_analysis.py` and runs automatically as the fourth analysis phase. It can also be run standalone:

```bash
MODEL=meta-llama/Meta-Llama-3-8B
DATA_DIR=datasets/meta-llama/Meta-Llama-3-8B

# Standalone (pre-edit)
python analysis/neuron_attribution.py \
    --model-name $MODEL --data-dir $DATA_DIR \
    --prompts two-hop first-hop second-hop \
    --subsets correct incorrect inconsistent

# Quick test
python analysis/neuron_attribution.py \
    --model-name $MODEL --data-dir $DATA_DIR \
    --prompts two-hop --subsets correct --max-examples 20
```

Outputs per `(subset, prompt_type)` under `{subset_dir}/neuron_attribution/{prompt_type}/`:
- `sublayer_lpi.csv`, `head_lpi.csv` -- per-example LPI
- `ffn_query_layers.csv`, `attn_query_layers.csv` -- residual stream decomposition
- `chain_profile.csv`, `chain_mean_layer.csv` -- aggregated chain attribution
- `chain_position_dist.csv` -- marginal position distribution

Under `{subset_dir}/neuron_attribution/`:
- `jaccard.csv`, `overlap_position.csv`, `overlap_position_by_layer.csv`

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
    correct:$DATA_DIR/composition_correct/neuron_attribution \
    incorrect:$DATA_DIR/composition_incorrect/neuron_attribution

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
        correct:$DATA_DIR/composition_correct/neuron_attribution \
        incorrect:$DATA_DIR/composition_incorrect/neuron_attribution

# Pre- vs post-edit comparison
python analysis/analyze_tables.py --subsets \
    pre-edit:$DATA_DIR/composition_incorrect \
    now_correct:$DATA_DIR/composition_incorrect_alphaedit/now_correct \
    still_incorrect:$DATA_DIR/composition_incorrect_alphaedit/still_incorrect
```

---

### Script reference

**Top-level orchestration:**

| Script | Purpose |
|--------|---------|
| `run_overnight.sh` | Full end-to-end pipeline: pre-edit + both methods + neuron attribution + HF push |
| `run_remaining.sh` | Re-run any failed steps (skips completed steps via existing output files) |
| `monitor.py` | Check pipeline progress: stats cache, step status, GPU, process |

**Analysis orchestrator** -- loads the model once and runs the full analysis pipeline:

| Script | Purpose |
|--------|---------|
| `run_analysis.py` | Unified orchestrator: pre-edit (evaluate + create subsets + analyses) or post-edit (edit + evaluate + split + analyses) via `--methods` flag |

**Analysis methods** -- called by the orchestrator; can also be run standalone:

| Script | Purpose |
|--------|---------|
| `generate_entity_description.py` | PatchScopes: patch hidden state at source position, decode entity description |
| `project_sublayer.py` | Sublayer projection: project attention/MLP output into vocabulary per layer |
| `knockout_attention.py` | Attention knockout: zero out attention from source to target, measure answer change |
| `evaluate_dataset.py` | Evaluate model on two_hop.csv (first-hop, second-hop, composition, shortcuts) |
| `neuron_attribution.py` | Neuron-level LPI and backward attribution chain; integrated into `run_analysis.py`, also runnable standalone (GPU required) |

**Analysis output scripts** -- read saved CSVs and produce tables or plots:

| Script | Purpose |
|--------|---------|
| `analyze_tables.py` | Build `layers.csv` from source CSVs (cached), print mechanism stage tables |
| `analyze_tables_neuron.py` | Print neuron attribution tables (LPI, chain, Jaccard, position) |
| `analyze_plots.py` | Generate mechanism plots (stage boxplots, PatchScopes heatmaps) |
| `analyze_plots_neuron.py` | Generate neuron attribution plots (LPI profiles, chain heatmaps, overlap bars) |

**Internal modules** -- imported by the scripts above, not run directly:

| Script | Purpose |
|--------|---------|
| `patch_model.py` | Monkey-patches GPT-2/Llama forward passes to capture internal states |
| `patched_generation.py` | Low-level generation with attention knockout and patching hooks; KV-cache variant gives ~32x speedup for PatchScopes |
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
| `create_subset.py` | Superseded -- logic inlined into `run_analysis.py` |
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

### MEMIT

MEMIT needs pre-computed covariance

Note: [legacy-datasets/wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) have samples with very long text. Meanwhile, our GPUs are 2x 3090. Consequently, computing the activations even for 1 sample but with 11k tokens will resulted in OutOfMemory error. So, we use `--batch_tokens 4096` to truncate the 11k tokens to maximum of 4096 tokens.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m EasyEdit.easyeditor.models.rome.layer_stats \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --apply_chat_template \
  --dataset wikipedia \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 \
  --layer_tmp model.layers.{layer_num}.mlp.down_proj \
  --to_collect mom2 \
  --sample_size 100000 \
  --batch_size 1 \
  --precision float32 \
  --stats_dir data/stats
```

### Others

AlphaEdit

```
python test_cake.py \
  --editing_method AlphaEdit \
  --sequential_editing \
  --model_type qwen3-4b-instruct \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --data_name MQuAKE-CF-3k-v2-clke
```

AlphaEdit_Circuit
```bash
python test_cake.py \
  --editing_method AlphaEdit_Circuit \
  --sequential_editing \
  --model_type qwen3-4b-instruct \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --data_name MQuAKE-CF-3k-v2-clke
```

CaKE
```bash
python test_cake.py \
  --editing_method CaKE \
  --sequential_editing \
  --model_type qwen3-4b-instruct \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --data_name MQuAKE-CF-3k-v2-clke
```

## Contributing

If you need to make changes to `EasyEdit` folder:

Change the submodule url in the `.gitmodules` file
```bash
git config -f .gitmodules submodule.EasyEdit.url "https://github.com/<your-name>/EasyEdit"
```

If you can't access GitHub directly, you can use [ghfast.top](https://ghfast.top) and GitHub's personal access token via `git remote` to write a local override in `.git/config` (not committed)
```bash
cd EasyEdit
git remote set-url origin https://user:your_token@ghfast.top/https://github.com/<your-name>/EasyEdit
```

Create new branch from the `main-cake` branch

```bash
cd EasyEdit
git checkout main-cake
git branch <your-branch>
git checkout <your-branch>
git push -u origin <your-branch>
```

Then, after you make changes, create 2 PRs: (1) PR to [jasonrichdarmawan/EasyEdit](https://github.com/jasonrichdarmawan/EasyEdit) repository, (2) PR to this repository.

Note: To avoid making changes to the submodule, but the code is not accessible for review, then any PR to this repository that require modified `EasyEdit` folder must be accompanied a PR to the [jasonrichdarmawan/EasyEdit](https://github.com/jasonrichdarmawan/EasyEdit) repository. Then PR to the [jasonrichdarmawan/EasyEdit](https://github.com/jasonrichdarmawan/EasyEdit) repository will be merged first, then the PR to this repository will make 1 additional commit to point to the `main-cake` branch.

## 🌻Acknowledgement

We thank for the project of [EasyEdit](https://github.com/zjunlp/EasyEdit), [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate).
The code in this work is built on top of these projects' codes.
