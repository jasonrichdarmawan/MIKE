from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pandas as pd

from datasets import Dataset
from analysis.patched_generation import generate_with_attention_knockout_all_layers
from analysis.utils import load_model, load_tokenizer, last_relation_word

from tqdm.auto import tqdm

def generate_answer_with_attention_knockout(
    entries,
    model, 
    tokenizer, 
    default_decoding: bool, 
    knockout_source: str,
    knockout_target: str, 
    k=0, 
    prompt="composition"
):
    """
    Example:
    - source_prompt: "the head of state of the country of citizenship of Stephen Harper is"
    - r1_template: "the country of citizenship of {}"
    - e1_label: "Stephen Harper"
    - r2_template: "the head of state of {}"
    - e2_label: "Canada"
    - if knockout_target == "e1":
      - generation_target_token_string: "the country of citizenship of Stephen Harper"
    - elif knockout_target == "e2" 
      - generation_target_token_string: "the head of state of Canada"
    - elif knockout_target == "r1"
      - generation_target_token_string: "citizenship"
    - elif knockout_target == "r2"
      - generation_target_token_string: "state"
    - if knockout_source == "e1":
      - knockout_source: "Stephen Harper"
    - elif knockout_source == "r1"
      - knockout_source: "citizenship"
    - elif knockout_source == "all":
      - knockout_source: all (inexplicitly handled in generate_with_attention_knock function)
    - elif knockout_source == "last":
      - knockout_source: "is" (inexplicitly handled in generate_with_attention_knock function)
    - if prompt == "composition":
      - source_prompt: "the head of state of the country of citizenship of Stephen Harper is"
    - elif prompt == "first-hop":
      - source_prompt: "the country of citizenship of Stephen Harper is"
    - elif prompt == "second-hop":
      - source_prompt: "the head of state of Canada is"
    """
    if knockout_target == "e1":
        generation_target_token_string = [t.format(e) for t, e in zip(entries["r1_template"], entries["e1_label"])]
    elif knockout_target == "e2":
        generation_target_token_string = [t.format(e) for t, e in zip(entries["r2_template"], entries["e2_label"])]
    elif knockout_target == "r1":
        generation_target_token_string = [last_relation_word(r1) for r1 in entries["r1_template"]]
    elif knockout_target == "r2":
        generation_target_token_string = [last_relation_word(r2) for r2 in entries["r2_template"]]
    else:
        raise ValueError(f"Knockout target {knockout_target} not supported")

    if knockout_source == "e1":
        knockout_source = entries["e1_label"]
    elif knockout_source == "r1":
        knockout_source = [last_relation_word(r1) for r1 in entries["r1_template"]]

    if prompt == "composition":
        source_prompt = entries["source_prompt"]
    elif prompt == "first-hop":
        source_prompt = [f"{r1_template.format(e1_label)} is" for r1_template, e1_label in
                         zip(entries["r1_template"], entries["e1_label"])]
    elif prompt == "second-hop":
        source_prompt = [f"{r2_template.format(e2_label)} is" for r2_template, e2_label in
                         zip(entries["r2_template"], entries["e2_label"])]
    else:
        raise ValueError(f"Prompt {prompt} not supported")

    tokenizer.padding_side = "left"
    generations = generate_with_attention_knockout_all_layers(model, tokenizer, source_prompt,
                                                              knockout_source, generation_target_token_string,
                                                              default_decoding, k)

    new_entries = {}
    for k in entries.keys():
        new_entries[k] = []
    new_entries["layer"] = []
    new_entries["generation"] = []
    for layer in generations.keys():
        for i in range(len(entries["source_prompt"])):
            for k, v in entries.items():
                new_entries[k].append(v[i])
            new_entries["layer"].append(layer)
            generation = generations[layer]
            new_entries["generation"].append(generation[i])

    return new_entries


def main(args):
    """
    The goal is to mask out attention weights from a source to a target position at a specific layer, causing these attention heads to contribute nothing to the model's prediction.

    If blocking a set of layers causes the model to no longer correctly predict the answer, we consider these layers to contain information critical to the prediction.

    Experiments:
    - knockout_source: "last", knockout_target: "e1", k: 3, prompt: "composition"
    """
    print(args)
    if not args.input_path:
        args.input_path = f"../datasets/{args.model_name}/two_hop.csv"
    dataset = pd.read_csv(args.input_path, index_col=0)

    model = load_model(args.model_name)
    model.eval()
    tokenizer = load_tokenizer(args.model_name)
    
    dataset = dataset.reset_index().rename(columns={"index": "id"})
    records = dataset.to_dict(orient="records")
    generation_frames = []
    for start in tqdm(range(0, len(records), args.batch_size), desc="Generating with attention knockout"):
        batch = records[start:start + args.batch_size]
        entries = {k: [row[k] for row in batch] for k in batch[0].keys()}
        generated = generate_answer_with_attention_knockout(
            entries=entries,
            model=model,
            tokenizer=tokenizer, 
            default_decoding=args.default_decoding,
            knockout_source=args.knockout_source,
            knockout_target=args.knockout_target,
            k=args.k,
            prompt=args.prompt
        )
        generation_frames.append(pd.DataFrame(generated))
    generations = pd.concat(generation_frames)
    generations = generations.sort_values(["id", "layer"])
    generations = generations.reset_index(drop=True)
    if not args.output_path:
        args.output_path = (f"../datasets/"
                            f"{args.model_name}/"
                            f"attention_knockout/"
                            f"{args.knockout_source}_{args.knockout_target}_attention_knockout_k{args.k}_{args.prompt}.csv")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(output_path, escapechar='\\')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    parser.add_argument("knockout_source", choices=["all", "last", "e1", "r1"])
    parser.add_argument("knockout_target", choices=["e1", "e2", "r1", "r2"])
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument('--default-decoding', action=BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--prompt", type=str, choices=["first-hop", "second-hop", "composition"],
                        default="composition")
    main(parser.parse_args())