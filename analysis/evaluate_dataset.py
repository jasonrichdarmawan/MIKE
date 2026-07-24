# %%

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from datasets import Dataset
from analysis.utils import _format_prompt, generate_and_test_answers, get_answers, _extract_answer_text_xml_tag, check_answer_in_pred

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm

# %%

def eval_composition(entries, model, tokenizer, base_model):
    """
    Example:
    - Source prompt: "the work that features the mother of Primrose Everdeen is"
    - Answers: ["The Hunger Games", "Hunger Games Saga", "The Hunger Games Saga"]
    """
    prompts = [_format_prompt(entry, tokenizer, base_model)
               for entry in entries["source_prompt"]]

    check_fn = lambda pred, answers: check_answer_in_pred(_extract_answer_text_xml_tag(pred, base_model), answers)
    return {
        "composition_correct": generate_and_test_answers(model, tokenizer, "composition", prompts, entries["id"], entries["e3_answers"], check_fn, base_model)
    }


def eval_first_hop(entries, model, tokenizer, base_model):
    """
    Example:
    - Source prompt: "the work that features the mother of Primrose Everdeen is"
    - Generated prompt: "the mother of Primrose Everdeen is"
    - Answers: ["Mrs. Everdeen", "Clara Everdeen"]
    """
    prompts = [_format_prompt(f"{r1_template.format(e1_label)} is", tokenizer, base_model)
               for r1_template, e1_label in zip(entries["r1_template"], entries["e1_label"])]

    check_fn = lambda pred, answers: check_answer_in_pred(_extract_answer_text_xml_tag(pred, base_model), answers)
    return {
        "first_hop_correct": generate_and_test_answers(model, tokenizer, "first hop", prompts, entries["id"], entries["e2_answers"], check_fn, base_model)
    }


def eval_second_hop(entries, model, tokenizer, base_model):
    """
    Example:
    - Source prompt: "the work that features the mother of Primrose Everdeen is"
    - Generated prompt: "the work that features Mrs. Everdeen is"
    - Answers: ["The Hunger Games", "Hunger Games Saga", "The Hunger Games Saga"]
    """
    prompts = [_format_prompt(f"{r2_template.format(e2_label)} is", tokenizer, base_model)
               for r2_template, e2_label in zip(entries["r2_template"], entries["e2_label"])]

    check_fn = lambda pred, answers: check_answer_in_pred(_extract_answer_text_xml_tag(pred, base_model), answers)
    return {
        "second_hop_correct": generate_and_test_answers(model, tokenizer, "second hop", prompts, entries["id"], entries["e3_answers"], check_fn, base_model)
    }


def eval_entity_shortcut(entries, model, tokenizer, base_model):
    r"""
    The goal is to filter out cases where no latent reasoing is performed. Specifically, where the model predicts due to a high correlation between :math:`e_1` and :math:`e_3`.
    
    Format: :math:`((e_1, "", e_2), (e_2, r_2, e_3))`
    
    Example:
    - Source prompt: ":math:`\underset{r_2}{\text{The spouse of}}` :math:`\underset{r_1}{\text{the performer of}}` :math:`\underset{e_1}{\text{Imagine}}` is"
    - Generated prompt: "The spouse of Imagine is"
    - Expected answer: ":math:`\underset{e_3}{\text{Yoko Ono}}`"
    """
    prompts = [_format_prompt(f"{r2_template.format(e1_label)} is", tokenizer, base_model)
               for r2_template, e1_label in zip(entries["r2_template"], entries["e1_label"])]

    check_fn = lambda pred, answers: check_answer_in_pred(_extract_answer_text_xml_tag(pred, base_model), answers)
    return {
        "entity_shortcut_correct": generate_and_test_answers(model, tokenizer, "entity shortcut", prompts, entries["id"], entries["e3_answers"], check_fn, base_model)
    }


def eval_relation_shortcut(entries, model, tokenizer, base_model):
    r"""
    The goal is to filter out cases where no latent reasoning is performed. Specifically, the model predicts generally popular entities.

    Format: :math:`(("". r_1, e_2), (e_2, r_2, e_3))`

    Example:
    - Source prompt: ":math:`\underset{r_2}{\text{The spouse of}}` :math:`\underset{r_1}{\text{the performer of}}` :math:`\underset{e_1}{\text{Imagine}}` is"
    - Generated prompt: "The spouse of the performer"
    - Expected answer: ":math:`\underset{e_3}{\text{Yoko Ono}}`"

    Caveat: 
    - Biran et al. [1] wrote "The spouse of the performer is", while the original source code generates "The spouse of the performer".
    - Biran et al. [1] remove with `[:-4]`. However, mot every `r1_template` ends with "... of {}". Consequently, removing `[:-4]` will cut off "the work that features {}" into "the work that featu".

    [1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
    """
    prompts = [_format_prompt(f"{r2_template.format(r1_template.format(''))}"[:-4], tokenizer, base_model)
               for r2_template, r1_template in zip(entries["r2_template"], entries["r1_template"])]

    check_fn = lambda pred, answers: check_answer_in_pred(_extract_answer_text_xml_tag(pred, base_model), answers)
    return {
        "relation_shortcut_correct": generate_and_test_answers(model, tokenizer, "relation shortcut", prompts, entries["id"], entries["e3_answers"], check_fn, base_model)
    }


def main(args):
    """
    The goal of `eval_first_hop`, `eval_second_hop` and `eval_composition` functions are to generate two dataset subsets per model:
    - The first subset is made up of cases where the model correctly answers both the two-hop query and the first hop.
    - The second subset is made up of cases where the model correctly answers both the first and second hop in isolation, but fails to answer the full two-hop query.

    Biran et al. [1] focus on the second subset because they are interested in understanding why answering the two-hop query fails despite the model "knowing" the two separate facts. Biran et al. [1] show this failure is likely due to the first fact being resolved too late.

    [1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
    """
    dataset = pd.read_csv(args.input_path, index_col=0)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.map(get_answers, fn_kwargs={"key": "e2"})
    dataset = dataset.map(get_answers, fn_kwargs={"key": "e3"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="cuda",
        dtype=getattr(torch, args.dtype),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Evaluate in Python to avoid pickling the model/tokenizer for Dataset.map
    n = len(dataset)
    first_hop_res = [None] * n
    second_hop_res = [None] * n
    composition_res = [None] * n
    entity_shortcut_res = [None] * n
    relation_shortcut_res = [None] * n

    for i in tqdm(range(0, n, args.batch_size), desc="Evaluating dataset"):
        end = min(i + args.batch_size, n)
        indices = list(range(i, end))
        batch = dataset.select(indices)

        r1 = eval_first_hop(batch, model, tokenizer, args.base_model)["first_hop_correct"]
        r2 = eval_second_hop(batch, model, tokenizer, args.base_model)["second_hop_correct"]
        comp = eval_composition(batch, model, tokenizer, args.base_model)["composition_correct"]
        esc = eval_entity_shortcut(batch, model, tokenizer, args.base_model)["entity_shortcut_correct"]
        rsc = eval_relation_shortcut(batch, model, tokenizer, args.base_model)["relation_shortcut_correct"]

        for idx, v1, v2, v3, v4, v5 in zip(indices, r1, r2, comp, esc, rsc):
            first_hop_res[idx] = v1
            second_hop_res[idx] = v2
            composition_res[idx] = v3
            entity_shortcut_res[idx] = v4
            relation_shortcut_res[idx] = v5

    dataset = dataset.add_column("first_hop_correct", first_hop_res)
    dataset = dataset.add_column("second_hop_correct", second_hop_res)
    dataset = dataset.add_column("composition_correct", composition_res)
    dataset = dataset.add_column("entity_shortcut_correct", entity_shortcut_res)
    dataset = dataset.add_column("relation_shortcut_correct", relation_shortcut_res)

    dataset_path = Path(args.input_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset.to_pandas()
    dataset = dataset.drop(["e2_answers", "e3_answers"], axis=1)
    dataset = dataset.reset_index(drop=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path)


def parse_args():
    BASE_MODELS = [
        "gpt2", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B",
        "Qwen/Qwen3.5-4B-Base"
    ]
    CHAT_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen3.5-4B", "Qwen/Qwen3-4B-Instruct-2507", "CohereLabs/tiny-aya-global",
        "CohereLabs/aya-expanse-8b",
    ]
    
    MODELS = BASE_MODELS + CHAT_MODELS
    
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=MODELS)
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    
    args.base_model = True if args.model_name in BASE_MODELS else False

    if not args.input_path:
        args.input_path = "datasets/two_hop.csv"
    if not args.output_path:
        filename = args.input_path.split("/")[-1]
        folder_path = "/".join(args.input_path.split("/")[:-1])
        args.output_path = f"{folder_path}/{args.model_name}/{filename}"
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
