from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from datasets import Dataset
from utils import load_model, load_tokenizer, generate_and_test_answers, get_answers


def eval_composition(entries, model, tokenizer):
    """
    Example:
    - Source prompt: "the work that features the mother of Primrose Everdeen is"
    - Answers: ["The Hunger Games", "Hunger Games Saga", "The Hunger Games Saga"]
    """
    return {
        "composition_correct": generate_and_test_answers(model, tokenizer, "composition", entries["source_prompt"], entries["id"], entries["e3_answers"])
    }


def eval_first_hop(entries, model, tokenizer):
    """
    Example:
    - Source prompt: "the work that features the mother of Primrose Everdeen is"
    - Generated prompt: "the mother of Primrose Everdeen is"
    - Answers: ["Mrs. Everdeen", "Clara Everdeen"]
    """
    prompts = [f"{r1_template.format(e1_label)} is" for r1_template, e1_label in
               zip(entries["r1_template"], entries["e1_label"])]
    return {
        "first_hop_correct": generate_and_test_answers(model, tokenizer, "first hop", prompts, entries["id"], entries["e2_answers"])
    }


def eval_second_hop(entries, model, tokenizer):
    """
    Example:
    - Source prompt: "the work that features the mother of Primrose Everdeen is"
    - Generated prompt: "the work that features Mrs. Everdeen is"
    - Answers: ["The Hunger Games", "Hunger Games Saga", "The Hunger Games Saga"]
    """
    prompts = [f"{r2_template.format(e2_label)} is" for r2_template, e2_label in
               zip(entries["r2_template"], entries["e2_label"])]
    return {
        "second_hop_correct": generate_and_test_answers(model, tokenizer, "second hop", prompts, entries["id"], entries["e3_answers"])
    }


def eval_entity_shortcut(entries, model, tokenizer):
    r"""
    The goal is to filter out cases where no latent reasoing is performed. Specifically, where the model predicts due to a high correlation between :math:`e_1` and :math:`e_3`.
    
    Format: :math:`((e_1, "", e_2), (e_2, r_2, e_3))`
    
    Example:
    - Source prompt: ":math:`\underset{r_2}{\text{The spouse of}}` :math:`\underset{r_1}{\text{the performer of}}` :math:`\underset{e_1}{\text{Imagine}}` is"
    - Generated prompt: "The spouse of Imagine is"
    - Expected answer: ":math:`\underset{e_3}{\text{Yoko Ono}}`"
    """
    prompts = [f"{r2_template.format(e1_label)} is" for r2_template, e1_label in
               zip(entries["r2_template"], entries["e1_label"])]
    return {
        "entity_shortcut_correct": generate_and_test_answers(model, tokenizer, "entity shortcut", prompts, entries["id"], entries["e3_answers"])
    }


def eval_relation_shortcut(entries, model, tokenizer):
    r"""
    The goal is to filter out cases where no latent reasoning is performed. Specifically, the model predicts generally popular entities.

    Format: :math:`(("". r_1, e_2), (e_2, r_2, e_3))`

    Example:
    - Source prompt: ":math:`\underset{r_2}{\text{The spouse of}}` :math:`\underset{r_1}{\text{the performer of}}` :math:`\underset{e_1}{\text{Imagine}}` is"
    - Generated prompt: "The spouse of the performer"
    - Expected answer: ":math:`\underset{e_3}{\text{Yoko Ono}}`"

    Caveat: The Biran et al. [1] wrote "The spouse of the performer is", while the original source code generates "The spouse of the performer"

    [1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
    """
    prompts = [f"{r2_template.format(r1_template.format(''))}"[:-4] for r2_template, r1_template in
               zip(entries["r2_template"], entries["r1_template"])]
    return {
        "relation_shortcut_correct": generate_and_test_answers(model, tokenizer, "relation shortcut", prompts, entries["id"], entries["e3_answers"])
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

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    tokenizer.padding_side = "left"

    dataset = dataset.map(eval_first_hop, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_second_hop, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_composition, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_entity_shortcut, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})
    dataset = dataset.map(eval_relation_shortcut, batched=True, batch_size=args.batch_size,
                          fn_kwargs={"model": model, "tokenizer": tokenizer})

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
        "Qwen/Qwen3.5-4B"
    ]
    
    MODELS = BASE_MODELS + CHAT_MODELS
    
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=MODELS)
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    args.base_model = True if args.model_name in BASE_MODELS else False
    if not args.base_model:
        raise NotImplementedError("Chat models are not supported yet.")

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
