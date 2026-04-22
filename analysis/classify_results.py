import ast
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis.utils import get_answers, check_answer_in_pred


def find_layers_by_classification(classified_generations: pd.DataFrame, classification: str | bool, col_name: str, apply_min: bool):
    """
    Example:
    - classified_generations:
      "id" | "classification" | "layer"
      1    | "e1"             | 0
      1    | "e2"             | 15
      2    | "e3"             | 16
    - classification: "e2" / "e3"
    - col_name: "last_e2_entity_layer" / "last_e3_entity_layer"
    - source_layer_classifications:
      id | "e1"         | "e2"      | "e3"
      1  | [0]          | [15]      | NaN
      2  | NaN          | [0]       | [16]
    - if classification == "e2":
      - return:
        id | "last_e2_entity_layer"
        1  | [15]
        2  | [0]

    Caveats:
    - if classification is str:
      - knockout_attention
      - entity_description
    - if classification is bool:
      - sublayer_projection
      - activation_patching
    - if layer_col == "layer", then the .csv file from experiments:
      - sublayer_projection
      - knockout_attention
    - if layer_col == "source_layer", then the .csv file from experiments:
      - entity_description
      - activation_patching
    """
    if "layer" in classified_generations.columns:
        layer_col = "layer"
    else:
        layer_col = "source_layer"
    source_layer_classifications = classified_generations.groupby(
        ["id", "classification"])[layer_col].apply(list).unstack("classification").reset_index()

    if apply_min:
        source_layer_classifications[classification] = source_layer_classifications[classification].apply(
            lambda x: [np.inf] if x is np.nan else x
        )
        source_layer_classifications[col_name] = source_layer_classifications[classification].apply(min)
        source_layer_classifications[col_name] = source_layer_classifications[col_name].replace(np.inf, np.nan)
    else:
        if classification not in source_layer_classifications.columns:
            source_layer_classifications[classification] = np.nan
        source_layer_classifications[col_name] = source_layer_classifications[classification].apply(
            lambda x: list(dict.fromkeys(x)) if type(x) is not float else x
        )

    return source_layer_classifications[["id", col_name]]


def classify_by_vanilla(generations: pd.DataFrame, merged: pd.DataFrame):
    """
    Caveat:
    - True if:
      - both the vanilla generation and the modified generation produced the correct answer
      - or the generated text string is exactly identical to the vanilla output.
    """
    merged["answers"] = pd.Series(
        [get_answers(row, "e3")["e3_answers"] for row in tqdm(merged.to_dict("records"), desc="Getting answers for classification")],
        index=merged.index
    )
    merged["generation_correct"] = pd.Series(
        [check_answer_in_pred(row["generation"], row["answers"]) for row in tqdm(merged.to_dict("records"), desc="Classifying generation correctness")],
        index=merged.index
    )
    merged["generation_vanilla_correct"] = pd.Series(
        [check_answer_in_pred(row["generation_vanilla"], row["answers"]) for row in tqdm(merged.to_dict("records"), desc="Classifying vanilla generation correctness")],
        index=merged.index
    )
    return ((merged["generation_vanilla_correct"] & merged["generation_correct"]) |
            (generations["generation"] == merged["generation_vanilla"]))


def classify_attention_knockout(generations: pd.DataFrame, target=None):
    vanilla_generations = generations[generations["layer"] == -1]
    merged = pd.merge(generations, vanilla_generations, on=["id"], suffixes=("", "_vanilla"))
    return classify_by_vanilla(generations, merged)


def classify_sublayer_projection(generations: pd.DataFrame, target: str):
    """
    Caveat:
    - True if:
      - for "prediction_rank" target: the prediction rank is 0
      - for "e2_ranks" target: any of the e2 ranks is 0
    """
    if generations[target].dtype == np.int64:
        return generations[target].apply(lambda x: x == 0)
    else:
        return generations[target].apply(lambda x: any(r == 0 for r in ast.literal_eval(x)))


def classify_prediction_correct_row(row):
    if pd.isna(row["generation"]):
        return False
    generation = row["generation"]
    e3_answers = get_answers(row, "e3")["e3_answers"]
    if check_answer_in_pred(generation, e3_answers):
        return True
    return False


def classify_prediction_correct(generations: pd.DataFrame, target=None):
    return pd.Series(
        [classify_prediction_correct_row(row) for row in tqdm(generations.to_dict("records"), desc="Classifying prediction correctness")],
        index=generations.index
    )


def get_entity_classification_answers(row, key: str):
    return ([f": {a} " for a in get_answers(row, key)[f"{key}_answers"]] +
            [f": {a}," for a in get_answers(row, key)[f"{key}_answers"]] +
            [f": {a}." for a in get_answers(row, key)[f"{key}_answers"]] +
            [f": {a}:" for a in get_answers(row, key)[f"{key}_answers"]] +
            [f"{a}: " for a in get_answers(row, key)[f"{key}_answers"]])


def classify_entity_row(row):
    if pd.isna(row["generation"]):
        return "other"
    generation = row["generation"]
    e1_answers = get_entity_classification_answers(row, "e1")
    e2_answers = get_entity_classification_answers(row, "e2")
    e3_answers = get_entity_classification_answers(row, "e3")
    if check_answer_in_pred(generation, e1_answers):
        return "e1"
    elif check_answer_in_pred(generation, e2_answers):
        return "e2"
    elif check_answer_in_pred(generation, e3_answers):
        return "e3"
    else:
        return "other"


def classify_entity(generations: pd.DataFrame, target=None):
    return pd.Series(
        [classify_entity_row(row) for row in tqdm(generations.to_dict("records"), desc="Classifying entity")],
        index=generations.index
    )


def classify_generations(generations_path: str, classification_fn, target=None, prev_layers=False, use_cache=True):
    """
    Caveats:
    - use `pd.read_csv(..., escapechar='\\')` due to `generate_entity_description.py` save with `pd.to_csv(..., escapechar='\\')`
    - if "layer" in generations.columns, then the .csv file from experiments:
      - knockout_attention
    - if prev_layers is True:
      - activation_patching
    """
    generations_path = Path(generations_path)
    classified_generations_path = generations_path.parent / f"{generations_path.stem}_classified.csv"

    if use_cache and classified_generations_path.exists():
        generations = pd.read_csv(classified_generations_path, index_col=0)
        print(f"Loaded classified generations from {classified_generations_path}")

    else:
        try:
            generations = pd.read_csv(generations_path, index_col=0, dtype={"id": int})
        except ValueError:
            print(f"Fixing generations from {generations_path}")
            with open(generations_path, "rb") as f:
                content = f.read()
            content = content.replace(b"\xe3\x80\x82", b" ")
            content = content.replace(b"\x0d", b"")
            with open(generations_path, "wb") as f:
                f.write(content)
            generations = pd.read_csv(generations_path, index_col=0, dtype={"id": int})

        print(f"Loaded generations from {generations_path}")
        generations["classification"] = classification_fn(generations, target)
        print(f"Saved classified generations to {classified_generations_path}")
        if use_cache:
            generations.to_csv(classified_generations_path)

    if "layer" in generations.columns:
        generations = generations[generations["layer"] != -1]
    if prev_layers:
        generations.loc[generations["source_layer"] <= generations["target_layer"], "classification"] = "masked"
    return generations


def load_attention_knockouts(dataset_dir: str):
    ret = defaultdict(dict)
    for source in ["last", "e1", "r1", "r2"]:
        for target in ["e1", "r1", "r2"]:
            path = f"{dataset_dir}/attention_knockout/{source}_{target}_attention_knockout_k3_composition.csv"
            if Path(path).exists():
                generations = classify_generations(path, classify_attention_knockout)
                ret[source][target] = find_layers_by_classification(
                    generations,
                    False,
                    f"{source}_{target}_attention_knockout_layers",
                    False)
    return ret


def load_activation_patching(dataset_dir: str, case_name: str | None=None):
    generations = defaultdict(dict)
    layers = defaultdict(dict)
    for source in ["e1", "r1", "r2", "last"]:
        for target in ["composition", "second-hop", "original", "last"]:
            path = f"{dataset_dir}/activation_patching/{source}_{target}_activation_patching{f'_{case_name}' if case_name else ''}.csv"
            if Path(path).exists():
                generations[source][target] = classify_generations(path, classify_prediction_correct, prev_layers=True)
                layers[source][target] = find_layers_by_classification(generations[source][target],
                                                                       True,
                                                                       f"{source}_{target}_activation_patching_layer",
                                                                       False)
    return layers


def load_entity_description_generations(dataset_dir: str):
    """
    Caveat:
    - Biran et al. [1] entity description generaton experiment' require to classify generations from 3 csv files. However, 1 csv file size is approximately 9.05 GB.

    [1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
    """
    generations = {}
    layers = defaultdict(dict)
    for source in ["last", "e1"]:
        gens = []
        for seed in range(1):
            path = f"{dataset_dir}/entity_description/{source}_entity_description_s{seed}.csv"
            gens.append(classify_generations(path, classify_entity))
        generations[source] = pd.concat(gens).drop_duplicates(
            subset=["id", "source_layer", "target_layer", "classification"]).sort_values(
            ["id", "source_layer", "target_layer"])
        for target in ["e2", "e3"]:
            layers[source][target] = find_layers_by_classification(generations[source],
                                                                   target,
                                                                   f"{source}_{target}_entity_layer",
                                                                   False)
    return layers


def load_sublayer_projections(dataset_dir: str):
    """
    Caveat:
    - this function use `classify_generations(..., use_cache=False)` because this function calls `classify_sublayer_projection(...)` twice on the same source file but with different targets
    """
    layers = defaultdict(dict)
    for sublayer in ["attention", "mlp"]:
        path = f"{dataset_dir}/sublayer_projection/{sublayer}_projections.csv"
        if Path(path).exists():
            generations = classify_generations(path, classify_sublayer_projection, target="prediction_rank",
                                               use_cache=False)
            layers[sublayer]["prediction"] = find_layers_by_classification(generations,
                                                                           True,
                                                                           f"{sublayer}_prediction_projection_layers",
                                                                           False)
            generations = classify_generations(path, classify_sublayer_projection, target="e2_ranks", use_cache=False)
            layers[sublayer]["e2"] = find_layers_by_classification(generations,
                                                                   True,
                                                                   f"{sublayer}_e2_projection_layers",
                                                                   False)
    return layers


def main(args):
    """
    Caveat:
    - col "{source}_{target}_entity_layer": Layer where the target hop entity is decoded
      - e.g "e1_e2_entity_layer": Layer where the second hop entity is decoded when the activation of the first hop entity is patched in to another prompt.
    - col "{source}_{target}_knockout_layers": Layers where the target hop entity is no longer decoded after knocking out attention from the source hop entity position to the target hop entity position
      - e.g "last_e1_attention_knockout_layers": Layers where the first hop entity is no longer decoded after knocking out attention from the prompt's last token position to the first hop entity
    - col "{sublayer}_{target}_projection_layers": Layers where the target hop entity is decoded from the sublayer's projection
      - e.g "attention_e2_projection_layers": Layers where the second hop entity is decoded from the attention sublayer's projection
    - col "{source}_{target}_activation_patching_layer": Layer where the model produces the correct answer after patching in the activation from a prompt's source hop entity position to the target hop entity position
      - e.g "e1_original_activation_patching_layer": Layer where the model produces the correct answer after patching in the activation from a prompt's first hop entity position to the original prompt
      - e.g "last_original_activation_patching_layer": Layer where the model produces the correct answer after patching in the activation from a prompt's last token position to the original prompt
    """
    entity_description_layers = load_entity_description_generations(args.results_dir)
    attention_knockout_layers = load_attention_knockouts(args.results_dir)
    sublayer_projection_layers = load_sublayer_projections(args.results_dir)
    activation_patching_layers = load_activation_patching(args.results_dir)

    two_hop = pd.read_csv(f"{args.results_dir}/two_hop.csv", index_col=0)
    base = two_hop[["id", "composition_correct"]]
    df = base \
        .merge(entity_description_layers["e1"]["e2"], on="id", how="left") \
        .merge(entity_description_layers["e1"]["e3"], on="id", how="left") \
        .merge(entity_description_layers["last"]["e2"], on="id", how="left") \
        .merge(entity_description_layers["last"]["e3"], on="id", how="left") \
        .merge(activation_patching_layers["e1"]["original"], on="id", how="left") \
        .merge(activation_patching_layers["last"]["original"], on="id", how="left") \
        .merge(sublayer_projection_layers["attention"]["prediction"], on="id", how="left") \
        .merge(sublayer_projection_layers["attention"]["e2"], on="id", how="left") \
        .merge(sublayer_projection_layers["mlp"]["prediction"], on="id", how="left") \
        .merge(sublayer_projection_layers["mlp"]["e2"], on="id", how="left") \
        .merge(attention_knockout_layers["last"]["e1"], on="id", how="left")

    df.to_csv(f"{args.results_dir}/layers.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("results_dir")
    tqdm.pandas()
    main(parser.parse_args())
