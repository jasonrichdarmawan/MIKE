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

pip install zhipuai sniffio
```

## 📚Analyze
Download the wikidata for analysis from [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/blob/main/datasets/two_hop.csv) into `datasets`.

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

- E. Biran et al. [1]:
  - Run `analysis/create_subset.py` to create `correct`, `incorrect`, `inconsistent` subsets. It will generate `datasets/{model_name}/composition_{correct/incorrect/inconsistent}/two_hop.csv` files. Use it to do back-patching experiments.
  - Run `analysis/evaluate_dataset.py` to evaluate a model on the `datasets/two_hop.csv` dataset. It will generate `datasets/{model_name}/two_hop.csv` file. Use it for the subsequent experiments.
    ```bash
    python analysis/evaluate_dataset.py gpt2 --input_path datasets/two_hop.csv --output_path datasets/gpt2/two_hop.csv
    ```
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

## 🧐Editing

Just run the following commond:
```
run.sh
```

### MEMIT

MEMIT needs pre-computed covariance

Note: [legacy-datasets/wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) have samples with very long text. Meanwhile, our GPUs are 2x 3090. Consequently, computing the activations even for 1 sample but with 11k tokens will resulted in OutOfMemory error. So, we use `--batch_tokens 4096` to truncate the 11k tokens to maximum of 4096 tokens.

```bash
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
MODEL_NAME=CohereLabs/tiny-aya-global
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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

### Others

no_edit
```bash
MODEL_TYPE=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=CohereLabs/tiny-aya-global
python test_cake.py \
  --editing_method no_edit \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --generation_batch_size 128 \
  --glue_eval_steps 0 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke
```

LoRA
```bash
```

R-ROME
```bash
python test_cake.py \
  --editing_method R-ROME \
  --model_type qwen3-4b-instruct \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --save_edited_model
```

MeLLo
```bash
python test_cake.py \
  --editing_method MeLLo \
  --model_type Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 2 \
  --sample_size 4 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke
```

MeLLo_KV
```bash
python test_cake.py \
  --editing_method MeLLo_KV \
  --model_type Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke
```

WISE
```bash
MODEL_TYPE=qwen3-4b-instruct
MODEL_TYPE=tiny-aya-global
python test_cake.py \
  --editing_method WISE \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
  --main_eval_steps 0 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --save_edited_model
```

AlphaEdit_Circuit
```bash
MODEL_TYPE=tiny-aya-global
MODEL_TYPE=qwen3-4b-instruct
python test_cake.py \
  --editing_method AlphaEdit_Circuit \
  --model_type $MODEL_TYPE \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --generation_batch_size 64 \
  --sample_size 1000 \
  --main_eval_steps 1 \
  --data_name MQuAKE-CF-3k-v2-clke \
  --save_edited_model
```

CaKE
```bash
python test_cake.py \
  --editing_method CaKE \
  --model_type Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --eval_with_chat_template \
  --batch_size 32 \
  --sample_size 1000 \
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