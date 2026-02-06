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
conda install conda-forge::micromamba
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

micromamba install datasets==3.3.1 einops==0.4.0 gpustat==1.1 hydra-core==1.1.1 higher==0.2.1 importlib-metadata==6.3.0 matplotlib==3.5.1 nltk==3.6.5 numpy==1.22.1 omegaconf==2.1.1 pandas==1.4.0 pyyaml==6.0 scikit-learn==1.0.2 scipy==1.7.3 sentence-transformers==3.2.1 tokenizers 
```

Note: You can change the pytorch-gpu version depending on your CUDA version.
```
micromamba install pytorch-gpu==2.0.0
```

```
micromamba install tqdm==4.62.3 transformers==4.44.2 openai==0.27.9 peft==0.7.1 timm==0.9.7 iopath==0.1.10 opencv==4.8.0 fairscale==0.4.13 sentencepiece seaborn

micromamba install beartype jaxtyping

pip install zhipuai
```

## 📚Analyze
Download the wikidata for analysis from [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/blob/main/datasets/two_hop.csv) into `datasets`.

- E. Biran et al. [1]:
  - Run `analysis/evaluate_dataset.py` to evaluate a model on the `datasets/two_hop.csv` dataset. It will generate `datasets/{model_name}/two_hop.csv` file. Use it for the subsequent experiments.
  - Run `analysis/generate_entity_description.py` to do Patchscopes entity description experiments.
  - Run `analysis/project_sublayer.py` to do sublayer projection experiments.
  - Run `analysis/patch_activations.py` to do back-patching experiments.
  - Run `analysis/classify_results.py` to classify the experiments' results.
  - Run `analysis/analyze_resutls.py` to generate figures.
- Y. Yao et al. [2]:
  - Run `analysis/create_subset.py` to create `correct`, `incorrect`, `inconsistent` subsets. It will generate `datasets/{model_name}/{correct/incorrect/inconsistent}.csv` files. Use it to do back-patching experiments.
  - Run `analysis/analyze_subset.py` to compute the `backpatch_t1`, `backpatch_t2`, `t1_to_t2` success rate.

[1] E. Biran et al., "Hopping Too Late: Exploring Limitations of Large Language models on Multi-Hop Queries," in EMNLP 2024
[2] Y. Yao et al., "CaKe: Circuit-aware Editing Enables Generalizable Knowledge Learners," in EMNLP 2025

## 🧐Editing
Just run the following commond:
```
run.sh
```

## 🌻Acknowledgement

We thank for the project of [EasyEdit](https://github.com/zjunlp/EasyEdit), [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate).
The code in this work is built on top of these projects' codes.