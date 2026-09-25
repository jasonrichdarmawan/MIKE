<h3 align="center"> MIKE: A Mechanistic Interpretability Knowledge Editing Method for Large Language Models </h3>

## Table of Contents
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- 🧐[Analyze](#analyze)
- 📚[Editing](#editing)
- 🌻[Acknowledgement](#acknowledgement)

---

## Pre-computed covariance, checkpoints, model-specific two hop CF benchmarks, logs

You can download the pre-computed covariance [here](https://huggingface.co/datasets/clke-results/CaKE/tree/main/jason/data/stats), checkpoints [here](https://huggingface.co/datasets/clke-results/CaKE/tree/main/jason/checkpoints), model-specific two hop CF benchmarks [here](https://huggingface.co/datasets/clke-results/CaKE/tree/main/jason/datasets), and the logs [here](https://huggingface.co/datasets/clke-results/CaKE/tree/main/jason/output)

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

micromamba install einops==0.4.0 gpustat==1.1 hydra-core==1.1.1 higher==0.2.1 importlib-metadata==6.3.0 matplotlib==3.10.8 nltk==3.6.5 omegaconf==2.1.1 pandas==1.4.0 pyyaml==6.0 scikit-learn==1.0.2 tokenizers==0.19.1

pip install "numpy>=1.24,<2" scipy sentence-transformers==3.2.1

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.57.6

micromamba install tqdm==4.62.3 openai==2.24.0 peft==0.7.1 timm==0.9.7 iopath==0.1.10 sentencepiece seaborn

pip install opencv-python==4.8.0.76 fairscale==0.4.13

micromamba install beartype jaxtyping pytest

pip install zhipuai sniffio

micromamba install datasets==3.3.1
```

## 📚Analyze
Download the wikidata for analysis from [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/blob/main/datasets/two_hop.csv) into `datasets`.

- Constructing counterfactual benchmark from Two Hop [1] dataset:
  - Run `analysis/two_hop_cf_create.py` to create the counterfactual benchmark. It will generate `datasets/two_hop_cf_{swapping_strategy}.csv`.
    `--swapping_strategy`: The swapping strategy to generate counterfactuals, either swapping `e2` values or swapping `e3` values.
    ```bash
    python -m analysis.two_hop_cf_create --input_path datasets/two_hop.csv --swapping_strategy e2
    ```
  - Run `analysis/evaluate_dataset.py` to evaluate a model on the `datasets/two_hop_{swapping_strategy}.csv` dataset. It will generate `datasets/{model_name}/two_hop_cf_{swapping_strategy}.csv` file. Use it for the subsequent experiments.
    ```bash
    MODEL_TYPE=Qwen/Qwen3-4B-Instruct-2507
    MODEL_TYPE=CohereLabs/tiny-aya-global
    python -m analysis.evaluate_dataset $MODEL_TYPE --input-path datasets/two_hop_cf_e2.csv
    ```
  - Run `analysis/create_benchmark_cf.py` to create valid counterfactual benchmark. It will generate `datasets/{model_name}/two_hop_cf_benchmark_{swapping_strategy}.csv`
    ```bash
    python -m analysis.create_benchmark_cf --input_path datasets/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2.csv
    ```

- Constructing MQuAKE for multi-lingual knowledge editing benchmark:
  - Copy `datasets/MQuAKE-CF-3k-v2-cake.json` to `datasets/MQuAKE-CF-3k-v2-clke.json`
  - Run `analysis/llm_translate.py` to translate
  ```bash
  DATA_PATH="datasets/MQuAKE-CF-3k-v2-clke.json"
  MODEL="gpt-oss-120b"
  ENDPOINT=<YOUR_ENDPOINT>
  API_KEY=<YOUR_API_KEY>
  : 'CASE_ID=<START> <STOP>"'
  CASE_ID="1 1"
  LANGUAGE="zh"
  python analysis/llm_translate.py --dataset_path $DATA_PATH --model $MODEL --endpoint $ENDPOINT --api_key $API_KEY --case_id ${=CASE_ID} --language ${=LANGUAGE}
  ```
  - Run `analysis/llm_translate_validate.py` to validate translation
  ```bash
  DATASET_PATH="datasets/MQuAKE-CF-3k-v2-clke.json"
  MODEL="DeepSeek-V4-Pro"
  : 'stable identifier of MODEL'
  JUDGE_ID="deepseek-v4-pro"
  ENDPOINT=<YOUR_ENDPOINT>
  API_KEY=<YOUR_API_KEY>
  : 'CASE_ID=<START> <STOP>"'
  CASE_ID="1 1"
  LANGUAGE="de fr ja zh"
  logging-tee --level INFO python analysis/llm_translate_validate.py \
    --dataset-path $DATASET_PATH \
    --model $MODEL \
    --endpoint $ENDPOINT \
    --api-key $API_KEY \
    --case-id ${=CASE_ID} \
    --languages ${=LANGUAGE} \
    --max-workers 64 \
    --requests-per-minute 100
  ```
  - Run `analysis/llm_translate_statistics.py` to get translation validation statistics
  ```bash
  DATASET_PATH="datasets/MQuAKE-CF-3k-v2-clke.json"
  JUDGE_ID="deepseek-v4-pro"
  CASE_ID="1 1"
  python analysis/llm_translate_statistics.py \
    --dataset-path $DATASET_PATH \
    --case-id ${=CASE_ID} \
    --judge-id $JUDGE_ID
  ```
- Run `analysis/llm_translate_agreement.py` to get agreement score between two judges
  ```
  DATASET_PATH="datasets/MQuAKE-CF-3k-v2-clke.json"
  JUDGE_ID_A="deepseek-v4-pro"
  JUDGE_ID_B="kimi-k2.5"
  LANGUAGES="de fr ja zh"
  CASE_ID="1 1"
  python analysis/llm_translate_agreement.py \
    --dataset-path $DATASET_PATH \
    --judge-a $JUDGE_ID_A \
    --judge-b $JUDGE_ID_B \
    --case-id ${=CASE_ID}
  ```

- Ours:
  - Run `analysis/evaluate_dataset.py` to evaluate a model on the `datasets/two_hop_cf_e2.csv` dataset. It will generate `datasets/{model_name}/two_hop_cf_e2.csv` file. Use it for the subsequent experiments.
    ```bash
    export CUDA_VISIBLE_DEVICES=1

    MODEL_TYPE=CohereLabs/tiny-aya-global
    MODEL_TYPE=Qwen/Qwen3-4B-Insturct-2507

    MODEL_TYPE=Qwen/Qwen2.5-7B-Instruct
    MODEL_TYPE=CohereLabs/aya-expanse-8b

    python -m analysis.evaluate_dataset $MODEL_TYPE --input-path datasets/two_hop_cf_e2.csv
    ```
  - Run `analysis/create_benchmark_cf.py` to create the benchmark based on `datasets/{model_name}/two_hop_cf_e2.csv`. It will generate `datasets/{model_name}/two_hop_cf_benchmark_e2.csv`
    ```bash
    INPUT_PATH=datasets/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2.csv
    python analysis/create_benchmark_cf.py --input_path $INPUT_PATH
    ```
  - Run `run_two_hop.py`. It will edit the model, evaluate the model, do sublayer projections, patchscopes, and attention knockout. Then, it will analyze and generate figures.

    Note: set `MODEL_TYPE` to HuggingFace's model name when `EDITING_METHOD == "no_edit"`
    ```bash
    export CUDA_VISIBLE_DEVICES=0

    EDITING_METHOD=no_edit
    
    MODEL_TYPE=Qwen/Qwen3-4B-Instruct-2507
    MODEL_TYPE=CohereLabs/tiny-aya-global
    
    MODEL_TYPE=Qwen/Qwen2.5-7B-Instruct 
    MODEL_TYPE=CohereLabs/aya-expanse-8b

    python run_two_hop.py \
      --editing_method $EDITING_METHOD \
      --model_type $MODEL_TYPE \
      --dtype bfloat16 \
      --sample_size 1000 \
      --batch_size 32 \
      --eval_with_chat_template \
      --do_evaluate \
      --do_patchscopes \
      --do_sublayer_projections \
      --do_knockout_attention
    ```

    Note: `aya-expanse-8b` and `qwen2.5-7b-instruct` need more than 24 GB VRAM to work. There rest of the model can run under 24 GB VRAM. Remove `--max_memory` if you don't evaluate these two models.

    ```bash
    export CUDA_VISIBLE_DEVICES=0,1

    EDITING_METHOD=AlphaEdit
    EDITING_METHOD="AlphaEdit_1-rank"
    EDITING_METHOD="AlphaEdit_1-rank_compute-matched"
    EDITING_METHOD=AlphaEdit_Circuit
    EDITING_METHOD="AlphaEdit_Circuit-no_layer"
    EDITING_METHOD=Revised_AlphaEdit
    EDITING_METHOD="Revised_AlphaEdit-lower_compute"

    MODEL_TYPE=qwen3-4b-instruct
    MODEL_TYPE=tiny-aya-global

    MODEL_TYPE=qwen2.5-7b-instruct
    MODEL_TYPE=aya-expanse-8b

    python run_two_hop.py \
      --editing_method $EDITING_METHOD \
      --model_type $MODEL_TYPE \
      --dtype bfloat16 \
      --sample_size 1000 \
      --batch_size 32 \
      --save_edited_model \
      --do_edit \
      --save_edit_statistics \
      --eval_with_chat_template \
      --do_evaluate \
      --do_patchscopes \
      --do_sublayer_projections \
      --do_knockout_attention \
      --max_memory 0:8GiB,1:8GiB
    ```

    In case experiment fail but checkpoint was saved, then you can continue the experiment from the checkpoint instead of losing progress.

    ```bash
    LOAD_MODEL_PATH=checkpoints/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_20260724-000058/edit_num_0350_case_79415
    LOAD_MODEL_PATH=checkpoints/CohereLabs/tiny-aya-global/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_20260724-000254/edit_num_0246_case_80844
    LOAD_MODEL_PATH=checkpoints/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_compute-matched_20260724-000600/edit_num_0350_case_79415

    python run_two_hop.py \
      --editing_method $EDITING_METHOD \
      --model_type $MODEL_TYPE \
      --dtype bfloat16 \
      --sample_size 1000 \
      --batch_size 32 \
      --load_edited_model \
      --load_model_path $LOAD_MODEL_PATH \
      --eval_with_chat_template \
      --do_evaluate \
      --do_patchscopes \
      --do_sublayer_projections \
      --do_knockout_attention
    ```
- E. Biran et al. [1]:
  - Run `analysis/evaluate_dataset.py` to evaluate a model on the `datasets/two_hop.csv` dataset. It will generate `datasets/{model_name}/two_hop.csv` file. Use it for the subsequent experiments.
    ```bash
    python analysis/evaluate_dataset.py gpt2 --input-path datasets/two_hop.csv --output-path datasets/gpt2/two_hop.csv
    ```
  - Run `analysis/create_subset.py` to create `correct`, `incorrect`, `inconsistent` subsets. It will generate `datasets/{model_name}/composition_{correct/incorrect/inconsistent}/two_hop.csv` files. Use it to do back-patching experiments.
  - Run `analysis/generate_entity_description.py` to do Patchscopes entity description experiments.
    <details>
      <summary>GPT2 example</summary>
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets/${MODEL}
      SEED=0

      COMPOSITION=correct
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv
      
      SOURCE_TOKEN=e1
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/entity_description/$SOURCE_TOKEN_entity_description_s${SEED}.csv
      python analysis/generate_entity_description.py $MODEL $SOURCE_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH --target-prompt description --seed $SEED

      SOURCE_TOKEN=last
      OUTPUT_PATH=$DATA_DIR/composition_${COMPOSITION}/entity_description/${SOURCE_TOKEN}_entity_description_s${SEED}.csv
      python analysis/generate_entity_description.py $MODEL $SOURCE_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH --target-prompt description --seed $SEED

      COMPOSITION=incorrect
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv

      SOURCE_TOKEN=e1
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/entity_description/${SOURCE_TOKEN}_entity_description_s${SEED}.csv
      python analysis/generate_entity_description.py $MODEL $SOURCE_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH --target-prompt description --seed $SEED

      SOURCE_TOKEN=last
      OUTPUT_PATH=$DATA_DIR/composition_${COMPOSITION}/entity_description/${SOURCE_TOKEN}_entity_description_s${SEED}.csv
      python analysis/generate_entity_description.py $MODEL $SOURCE_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH --target-prompt description --seed $SEED
      ```
    </details>
  - Run `analysis/project_sublayer.py` to do sublayer projection experiments.
    <details>
      <summary>GPT2 example</summary>
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets/${MODEL}

      COMPOSITION=correct
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv

      SUBLAYER=attention
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/sublayer_projection/${SUBLAYER}_projections.csv
      python analysis/project_sublayer.py $MODEL $SUBLAYER --input-path $INPUT_PATH --output-path $OUTPUT_PATH
    
      SUBLAYER=mlp
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/sublayer_projection/${SUBLAYER}_projections.csv
      python analysis/project_sublayer.py $MODEL $SUBLAYER --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      COMPOSITION=incorrect
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv

      SUBLAYER=attention
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/sublayer_projection/${SUBLAYER}_projections.csv
      python analysis/project_sublayer.py $MODEL $SUBLAYER --input-path $INPUT_PATH --output-path $OUTPUT_PATH
    
      SUBLAYER=mlp
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/sublayer_projection/${SUBLAYER}_projections.csv
      python analysis/project_sublayer.py $MODEL $SUBLAYER --input-path $INPUT_PATH --output-path $OUTPUT_PATH
      ```
    </details>
  - Run `analysis/patch_activations.py` to do back-patching experiments.
    <details>
      <summary>GPT2 example</summary>
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets/${MODEL}

      COMPOSITION=correct
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv
    
      SOURCE_TOKEN=e1
      TARGET_TOKEN=original
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      SOURCE_TOKEN=last
      TARGET_TOKEN=original
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      COMPOSITION=incorrect
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv

      SOURCE_TOKEN=e1
      TARGET_TOKEN=original
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      SOURCE_TOKEN=last
      TARGET_TOKEN=original
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH
      ```
    </details>
  - Run `analysis/knockout_attention.py` to do attention knockout experiments.
    <details>
      <summary>GPT2 example</summary>
      ```bash
      MODEL=gpt2
      PROMPT_TYPE=composition
      DATA_DIR=datasets/${MODEL}
      K=3
      
      COMPOSITION=correct
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv
      
      KNOCKOUT_SOURCE=last
      KNOCKOUT_TARGET=e1
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/attention_knockout/${KNOCKOUT_SOURCE}_${KNOCKOUT_TARGET}_attention_knockout_k${K}_${PROMP_TYPE}.csv

      python analysis/knockout_attention.py $MODEL $KNOCKOUT_SOURCE $KNOCKOUT_TARGET --input-path $INPUT_PATH --output-path $OUTPUT_PATH --k $K --prompt $PROMPT_TYPE

      COMPOSITION=incorrect
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv
      
      KNOCKOUT_SOURCE=last
      KNOCKOUT_TARGET=e1
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/attention_knockout/${KNOCKOUT_SOURCE}_${KNOCKOUT_TARGET}_attention_knockout_k${K}_${PROMP_TYPE}.csv

      python analysis/knockout_attention.py $MODEL $KNOCKOUT_SOURCE $KNOCKOUT_TARGET --input-path $INPUT_PATH --output-path $OUTPUT_PATH --k $K --prompt $PROMPT_TYPE
      ```
    </details>
  - Run `analysis/classify_results.py` to classify the experiments' results.
    <details>
      <summary>GPT2 example</summary>
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets/${MODEL}

      COMPOSITION=correct
      RESULT_DIR=${DATA_DIR}/composition_${COMPOSITION}
      python analysis/classify_results.py $RESULT_DIR

      COMPOSITION=incorrect
      RESULT_DIR=${DATA_DIR}/composition_${COMPOSITION}
      python analysis/classify_results.py $RESULT_DIR
      ```
    </details>
  - Run `analysis/analyze_resutls.py` to generate figures.
    <details>
      <summary>GPT2 example</summary>
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets
      OUTPUT_PATH=output

      python analysis/analyze_results.py $DATA_DIR $OUTPUT_PATH -m $MODEL
      ```
    </details>
- Y. Yao et al. [2]:
  - Run `analysis/analyze_subset.py` to compute the `backpatch_t1`, `backpatch_t2`, `t1_to_t2` success rate.
    <details>
      <summary>GPT2 example</summary>

      do extra activation patching
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets/${MODEL}

      COMPOSITION=incorrect
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv

      SOURCE_TOKEN=e1
      TARGET_TOKEN=last
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      COMPOSITION=inconsistent
      INPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/two_hop.csv
    
      SOURCE_TOKEN=e1
      TARGET_TOKEN=original
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      SOURCE_TOKEN=last
      TARGET_TOKEN=original
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH

      SOURCE_TOKEN=e1
      TARGET_TOKEN=last
      OUTPUT_PATH=${DATA_DIR}/composition_${COMPOSITION}/activation_patching/${SOURCE_TOKEN}_${TARGET_TOKEN}_activation_patching.csv
      python analysis/patch_activations.py $MODEL $SOURCE_TOKEN $TARGET_TOKEN --input-path $INPUT_PATH --output-path $OUTPUT_PATH
      ```

      analyze
      ```bash
      MODEL=gpt2
      DATA_DIR=datasets/gpt2

      python analysis/analyze_subset.py --data_dir $DATA_DIR
      ```
    </details>

[1] E. Biran et al., "Hopping Too Late: Exploring Limitations of Large Language models on Multi-Hop Queries," in EMNLP 2024
[2] Y. Yao et al., "CaKe: Circuit-aware Editing Enables Generalizable Knowledge Learners," in EMNLP 2025

## Confidence interval

After you run `run_two_hop.py` or `test_cake.py`, you can use the following code to compute the confidence interval

```bash
METRICS_FOLDER_PATH=<METRICS_FOLDER_PATH>

./metrics_to_statistics.sh $METRICS_FOLDER_PATH
```

Use `--comparison-metrics-path` to compute the paired CI.

```bash
METRICS_FOLDER_PATH=<METRICS_FOLDER_PATH>
COMPARISON_METRICS_FOLDER_PATH=<COMPARISON_METRICS_FOLDER_PATH>
RES_FILE=<RES_FILE>

./metrics_to_statistics.sh $METRICS_FOLDER_PATH --comparison-metrics-path $COMPARISON_METRICS_FOLDER_PATH --res-path $RES_FILE
```

## String-based evaluator error rate

```bash
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# DATASET_NAME="Two-Hop"
# RESULTS_PATH="./output/Qwen/Qwen2.5-7B-Instruct/two_hop_cf_e2/sequential_s1000_s32/Revised_AlphaEdit_20260724-161909"
DATASET_NAME="MQuAKE-CF-3k-v2"
RESULTS_PATH="./output/Qwen/Qwen2.5-7B-Instruct/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/Revised_AlphaEdit_20260726_093055"
LANG="en"
python analysis/human_annotate/annotate.py \
  --model-name $MODEL_NAME \
  --dataset-name $DATASET_NAME \
  --results-path $RESULTS_PATH \
  --lang $LANG
```

```bash
RESULTS_PATH="./output/Qwen/Qwen2.5-7B-Instruct/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/Revised_AlphaEdit_20260726_093055"
LANG="zh"
python analysis/human_annotate/metrics.py \
  --results-path $RESULTS_PATH
```

## Compute metrics separately

```bash
EXPERIMENT_ROOT=output/CohereLabs/aya-expanse-8b/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/Revised_AlphaEdit_20260726_093849
python output_analysis/metrics_question_cloze.py \
  --experiment-root $EXPERIMENT_ROOT
```

### Covariance

`ROME`, `R-ROME`, `MEMIT`, `IFMET`, `AlphaEdit`, `AlphaEdit_Circuit`, `Revised_AlphaEdit` needs a pre-computed covariance

Note: [legacy-datasets/wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) have samples with very long text. Meanwhile, our GPUs are 2x 3090. Consequently, computing the activations even for 1 sample but with 11k tokens will resulted in OutOfMemory error. So, we use `--batch_tokens 4096` to truncate the 11k tokens to maximum of 4096 tokens.

Note 2: Pre-computing the covariance will take approximately 23 hours on 2x 3090. You can download [here](https://huggingface.co/datasets/clke-results/CaKE/tree/main/jason/data/stats) to save time.

```bash
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
MODEL_NAME=CohereLabs/tiny-aya-global

python -m EasyEdit.easyeditor.models.rome.layer_stats \
  --model_name $MODEL_NAME \
  --apply_chat_template \
  --dataset wikipedia \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 \
  --layer_tmp model.layers.{layer_num}.mlp.down_proj \
  --to_collect mom2 \
  --sample_size 100000 \
  --batch_size 1 \
  --batch_tokens 4096 \
  --precision float32 \
  --stats_dir data/stats
```

Note 3: If your setup have limited VRAM, test whether the GPU can fit. If you use multi-GPU, adjust the `--max_memory` flag to fit model weights, activations and covariance into the VRAM. Then, remove the `--fake_samples` and `--fake_sequence_length` after test. It's better to offload some parameters to the disk than receiving Out-of-Memory (OOM) error.

```bash
export CUDA_VISIBLE_DEVICES=1,4

MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MODEL_NAME=CohereLabs/aya-expanse-8b

python -m EasyEdit.easyeditor.models.rome.layer_stats \
  --model_name $MODEL_NAME \
  --apply_chat_template \
  --dataset wikipedia \
  --layers 4,5,6,7,8 \
  --layer_tmp model.layers.{layer_num}.mlp.down_proj \
  --to_collect mom2 \
  --sample_size 100000 \
  --batch_size 1 \
  --batch_tokens 4096 \
  --precision float32 \
  --stats_dir data/stats \
  --max_memory 0:11GiB,1:11GiB \
  --fake_samples 100 \
  --fake_sequence_length 4096
```

### Others

no_edit
```bash
export CUDA_VISIBLE_DEVICES=4

EDITING_METHOD=no_edit

MODEL_TYPE=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=CohereLabs/tiny-aya-global

MODEL_TYPE=Qwen/Qwen2.5-7B-Instruct
MODEL_TYPE=CohereLabs/aya-expanse-8b

python test_cake.py \
  --editing_method $EDITING_METHOD \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --load_model \
  --sample_size 1000 \
  --batch_size 32 \
  --generation_batch_size 128 \
  --glue_eval_steps 0 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --max_memory 0:8GiB,1:8GiB
```

`ROME`, `R-ROME`, `MEMIT`, `IFMET`, `WISE`
```bash
EDITING_METHOD=ROME
EDITING_METHOD=R-ROME
EDITING_METHOD=MEMIT
EDITING_METHOD=IFMET
EDITING_METHOD=WISE
EDITING_METHOD=ENCORE

MODEL_TYPE=qwen3-4b-instruct
MODEL_TYPE=tiny-aya-global

MODEL_TYPE=qwen2.5-7b-instruct
MODEL_TYPE=aya-expanse-8b

python test_cake.py \
  --editing_method $EDITING_METHOD \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --load_model \
  --do_edit \
  --eval_with_chat_template \
  --sample_size 1000 \
  --batch_size 32 \
  --generation_batch_size 128 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --save_edited_model \
  --max_memory 0:6GiB,1:10GiB
```

`AlphaEdit`, `AlphaEdit_Circuit`, `Revised_AlphaEdit`, `AlphaEdit 1-rank`

Note: save statistics
```bash
export CUDA_VISIBLE_DEVICES=1,2

EDITING_METHOD=AlphaEdit
EDITING_METHOD="AlphaEdit_1-rank"
EDITING_METHOD="AlphaEdit_1-rank_compute-matched"
EDITING_METHOD=AlphaEdit_Circuit
EDITING_METHOD="AlphaEdit_Circuit-no_layer"
EDITING_METHOD=Revised_AlphaEdit

MODEL_TYPE=qwen3-4b-instruct
MODEL_TYPE=tiny-aya-global

MODEL_TYPE=qwen2.5-7b-instruct
MODEL_TYPE=aya-expanse-8b

python test_cake.py \
  --editing_method $EDITING_METHOD \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --load_model \
  --do_edit \
  --eval_with_chat_template \
  --sample_size 1000 \
  --batch_size 32 \
  --generation_batch_size 128 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --save_edit_statistics \
  --save_edited_model \
  --max_memory 0:6GiB,1:10GiB
```

`MeLLo` and `MeLLo_KV`
```bash
EDITING_METHOD=MeLLo
EDITING_METHOD=MeLLo_KV

MODEL_TYPE=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=CohereLabs/tiny-aya-global
python test_cake.py \
  --editing_method $EDITING_METHOD \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --load_model \
  --eval_with_chat_template \
  --sample_size 1000 \
  --batch_size 32 \
  --generation_batch_size 128 \
  --main_eval_steps 0 \
  --glue_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke
```

`CaKE`
```bash
EDITING_METHOD=CaKE

MODEL_TYPE=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=CohereLabs/tiny-aya-global
python test_cake.py \
  --editing_method $EDITING_METHOD \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --load_model \
  --do_edit \
  --eval_with_chat_template \
  --sample_size 1000 \
  --batch_size 32 \
  --generation_batch_size 128 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --save_edited_model
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
