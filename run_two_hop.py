import argparse
import json
import os

import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Currently testing AlphaEdit
from EasyEdit.easyeditor.models.alphaedit import AlphaEditHyperParams

from EasyEdit.easyeditor.util.alg_dict import ALG_DICT
from EasyEdit.easyeditor.util import nethook
from edit_utils import compute_edit_quality, edit
from test_cake import calculate_averages


def _load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_list(values):
    return json.dumps(values or [])


def _normalize_answers(canonical, aliases=None):
    aliases = aliases or []
    answers = list(aliases)
    if canonical not in answers:
        answers.append(canonical)
    return answers


def _build_entity_metadata(item):
    e1 = item["requested_rewrite"]["subject"]
    e2 = item["new_single_hops"][0]["answer"]
    e3 = item["new_answer"]

    e1_answers = [e1]
    e2_aliases = item["new_single_hops"][0].get("answer_alias", []) or []
    e3_aliases = item.get("new_answer_alias", []) or []
    e2_answers = _normalize_answers(e2, e2_aliases)
    e3_answers = _normalize_answers(e3, e3_aliases)

    return {
        "id": item["case_id"],
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e1_answers": _json_list(e1_answers),
        "e2_answers": _json_list(e2_answers),
        "e3_answers": _json_list(e3_answers),
        # Keep both schema variants for compatibility with analysis helpers.
        "e1_label": e1,
        "e2_label": e2,
        "e3_label": e3,
        "e1_aliases": _json_list([]),
        "e2_aliases": _json_list(e2_aliases),
        "e3_aliases": _json_list(e3_aliases),
    }


def append_two_hop_row(two_hop_rows, item, composition_correct):
    row = _build_entity_metadata(item)
    row["composition_correct"] = bool(composition_correct)
    two_hop_rows.append(row)


def append_entity_description_rows(entity_description_rows, entity_result, item):
    meta = _build_entity_metadata(item)
    for i, generation in enumerate(entity_result.get("generation", [])):
        entity_description_rows.append({
            "id": meta["id"],
            "source_prompt": entity_result["source_prompt"][i],
            "source_layer": entity_result["source_layer"][i],
            "target_layer": entity_result["target_layer"][i],
            "generation": generation,
            "e1": meta["e1"],
            "e2": meta["e2"],
            "e3": meta["e3"],
            "e1_answers": meta["e1_answers"],
            "e2_answers": meta["e2_answers"],
            "e3_answers": meta["e3_answers"],
            "e1_label": meta["e1_label"],
            "e2_label": meta["e2_label"],
            "e3_label": meta["e3_label"],
            "e1_aliases": meta["e1_aliases"],
            "e2_aliases": meta["e2_aliases"],
            "e3_aliases": meta["e3_aliases"],
        })


def append_knockout_rows(attention_knockout_rows, knockout_result, item):
    meta = _build_entity_metadata(item)
    layers = knockout_result.get("layer", [])
    generations = knockout_result.get("generation", [])

    has_vanilla = any(layer == -1 for layer in layers)
    for i, layer in enumerate(layers):
        attention_knockout_rows.append({
            "id": meta["id"],
            "layer": layer,
            "generation": generations[i],
            "e3": meta["e3"],
            "e3_answers": meta["e3_answers"],
            "e3_label": meta["e3_label"],
            "e3_aliases": meta["e3_aliases"],
        })

    # Compatibility fallback: add vanilla row if helper output does not include layer -1.
    if not has_vanilla:
        attention_knockout_rows.append({
            "id": meta["id"],
            "layer": -1,
            "generation": "",
            "e3": meta["e3"],
            "e3_answers": meta["e3_answers"],
            "e3_label": meta["e3_label"],
            "e3_aliases": meta["e3_aliases"],
        })


def append_projection_rows(projection_rows, proj_result, item):
    meta = _build_entity_metadata(item)
    row_count = len(proj_result.get("layer", []))
    for i in range(row_count):
        projection_rows.append({
            "id": meta["id"],
            "layer": proj_result["layer"][i],
            "top_k_words": _json_list(proj_result["top_k_words"][i]),
            "top_k_probs": _json_list(proj_result["top_k_probs"][i]),
            "e2_ranks": _json_list(proj_result["e2_ranks"][i]),
            "e3_ranks": _json_list(proj_result["e3_ranks"][i]),
            "prediction_rank": proj_result["prediction_rank"][i],
        })


from analysis.project_sublayer import project_attention
def run_project_sublayer(model, tokenizer, item, target):

    # the first new single hop question is (e1, r1) and its ans answer is e2_new.
    bridge = item["new_single_hops"][0]
    entries = {
        "id": [item["case_id"]],

        # TODO: CHECK CORRECTNEESS OF CHOICE
        # cloze question is the source prompt from two_hop.csv
        "source_prompt": [item["cloze_question"]],
        "e2_answers": [(bridge.get("answer_alias", []) or []) + [bridge["answer"]]],
        "e3_answers": [(item.get("new_answer_alias", []) or []) + [item["new_answer"]]],
    }
    return project_attention(entries, model, tokenizer, target=target)


# patchscope
# 
from analysis.patched_generation import get_hidden_states, generate_with_patching_all_layers
from analysis.utils import get_layer_names
def run_entity_description_e2_new(model, tokenizer, item, do_sample=False):
    # Replicate generate_entity_description_with_patching(source="e1")
    source_prompt =  item["cloze_question"]
    
    target_prompt = (
        "Syria: Syria is a country in the Middle East, "
        "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, "
        "Samsung: Samsung is a South Korean multinational corporation, "
        "x"
    )

    entries = {
        "id": [item["case_id"]],
        "source_prompt": [source_prompt],
        "r1_template": [item["single_hops"][0]["cloze"].replace(" is", "")],
        "e1_label": item["requested_rewrite"]["subject"],
        "target_prompt": [target_prompt],
    }

    entities = [t.format(e) for t, e in zip(entries["r1_template"], entries["e1_label"])]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    hidden_states = get_hidden_states(model, tokenizer, entities, entries["source_prompt"])
    tokenizer.padding_side = "left"
    generations = generate_with_patching_all_layers(
        model,
        tokenizer,
        hidden_states,
        entries["target_prompt"],
        "x",
        do_sample,
    )
    tokenizer.padding_side =  old_padding_side

    entry_count = len(entries["source_prompt"])
    layer_count = len(get_layer_names(model))
    new_entries = {k: [] for k in entries.keys()}
    new_entries["source_layer"] = []
    new_entries["target_layer"] = []
    new_entries["generation"] = []

    for source_layer in range(layer_count):
        for target_layer in range(layer_count):
            for i in range(entry_count):
                for k, v in entries.items():
                    new_entries[k].append(v[i])
                new_entries["source_layer"].append(source_layer)
                new_entries["target_layer"].append(target_layer)
                new_entries["generation"].append(generations[source_layer, target_layer][i])

    return new_entries

from analysis.knockout_attention import generate_answer_with_attention_knockout
from analysis.patched_generation import get_hidden_states, generate_with_patching_all_layers
from analysis.utils import get_layer_names
def run_knockout_bridge(model, tokenizer, item, knockout_source, knockout_target, k=3, prompt="composition", default_decoding=False):
    # Create entries for knockout attention
    data = {
        "id": [item["case_id"]],
        "source_prompt": [item["cloze_question"]],

        # e2_old 
        # "e1_label": [item["single_hops"][0]["answer"]],  
        "e1_label": [item["requested_rewrite"]["subject"]],  
        # e3_old
        # "e2_label": [item["single_hops"][1]["answer"]],
        # e2_new
        # "e2_label": [item["single_hops"][0]["answer"]],

        "r2_template": [item["single_hops"][1]["cloze"].replace(" is", "")],  # Extract template
        "r1_template": [item["single_hops"][0]["cloze"].replace(" is", "")],  # Extract template
    }
    df = pd.DataFrame(data)
    
    entries = {
        "id": df["id"].tolist(),
        "source_prompt": df["source_prompt"].tolist(),
        "r1_template": df["r1_template"].tolist(),
        "e1_label": df["e1_label"].tolist(),
        "r2_template": df["r2_template"].tolist(),
        # "e2_label": df["e2_label"].tolist(),
    }
    
    result = generate_answer_with_attention_knockout(entries, model, tokenizer, default_decoding, 
                                                      knockout_source, knockout_target, k, prompt)
    return result



def main():
    parser = argparse.ArgumentParser(description="Run AlphaEdit in dataset order: edit rows are edited, propagated rows are evaluated.")
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--metrics_save_dir", default="./output", type=str)
    parser.add_argument("--max_rows", default=None, type=int)
    args = parser.parse_args()

    hparams = AlphaEditHyperParams.from_hparams(f"./EasyEdit/hparams/AlphaEdit/{args.model_type}.yaml")
    alg_name = hparams.alg_name
    apply_algo = ALG_DICT[alg_name]

    model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    cases = _load_cases(args.dataset_path)
    if args.max_rows is not None:
        cases = cases[: args.max_rows]

    n_edit = sum(1 for x in cases if x.get("row_role") == "edit")
    n_propagated = sum(1 for x in cases if x.get("row_role") == "propagated_eval")
    print(f"Loaded {len(cases)} rows | edit={n_edit} propagated_eval={n_propagated}")

    # for metrics stored in json, as done in test_cake
    all_metrics = []
    
    # used to generate seperate csvs for different analyses.
    two_hop_rows = []
    entity_description_rows = []
    attention_knockout_rows = []
    attention_proj_rows = []
    mlp_proj_rows = []

    skipped = 0
    edit_count = 0
    prop_eval_count = 0

    for row_index, item in enumerate(tqdm(cases, desc="Running rows in order")):
        row_role = item.get("row_role")
        if row_role == "edit":
            edit_count += 1
            model, metrics = edit(
                model,
                tokenizer,
                item,
                hparams,
                alg_name,
                apply_algo,
                test_generation=False,
                restore_weights=False,
            )
            metrics["row_role"] = "edit"
            metrics["row_index"] = row_index

        elif row_role == "propagated_eval":
            metrics = {
                "case_id": item["case_id"],
                "requested_rewrite": item.get("requested_rewrite", []),
                "time": 0,
                "post": compute_edit_quality(model, tokenizer, item, hparams, test_generation=False),
                "row_role": "propagated_eval",
                "row_index": row_index,
            }
            prop_eval_count += 1
        else:
            skipped += 1
            continue

        # Check if e3_new is successfully predicted
        composition_correct = False
        if "post" in metrics and "accuracy" in metrics["post"]:
            if len(metrics["post"]["accuracy"]) > 1:
                composition_correct = metrics["post"]["accuracy"][0] or metrics["post"]["accuracy"][1]
            elif len(metrics["post"]["accuracy"]) == 1:
                composition_correct = metrics["post"]["accuracy"][0] 
        
        metrics["composition_correct"] = composition_correct
        append_two_hop_row(two_hop_rows, item, composition_correct)

        if composition_correct:
            # sublayer projection
            for target in ["attention", "mlp"]:
                try:
                    proj_result = run_project_sublayer(model, tokenizer, item, target)
                    if target == "attention":
                        append_projection_rows(attention_proj_rows, proj_result, item)
                    else:
                        append_projection_rows(mlp_proj_rows, proj_result, item)
                except Exception:
                    pass

            # entity description probing for e2_new decodability
            try:
                entity_result = run_entity_description_e2_new(model, tokenizer, item, do_sample=False)
                append_entity_description_rows(entity_description_rows, entity_result, item)
            except Exception:
                pass

            # knockout bridge representation
            knockout_source = "last"
            knockout_target = "e1"
            try:
                knockout_result = run_knockout_bridge(model, tokenizer, item, knockout_source, knockout_target, 
                                                        k=3, prompt="composition", default_decoding=False)
                append_knockout_rows(attention_knockout_rows, knockout_result, item)
            except Exception:
                pass
        
        all_metrics.append(metrics)

    os.makedirs(args.metrics_save_dir, exist_ok=True)

    two_hop_df = pd.DataFrame(two_hop_rows)
    if not two_hop_df.empty:
        two_hop_df = two_hop_df.drop_duplicates("id")
    two_hop_df.to_csv(f"{args.metrics_save_dir}/two_hop.csv", index=False)

    entity_description_dir = f"{args.metrics_save_dir}/entity_description"
    attention_knockout_dir = f"{args.metrics_save_dir}/attention_knockout"
    sublayer_projection_dir = f"{args.metrics_save_dir}/sublayer_projection"
    os.makedirs(entity_description_dir, exist_ok=True)
    os.makedirs(attention_knockout_dir, exist_ok=True)
    os.makedirs(sublayer_projection_dir, exist_ok=True)

    pd.DataFrame(entity_description_rows).to_csv(
        f"{entity_description_dir}/e1_entity_description_s0.csv", index=False
    )
    pd.DataFrame(attention_knockout_rows).to_csv(
        f"{attention_knockout_dir}/last_e1_attention_knockout_k3_composition.csv", index=False
    )
    pd.DataFrame(attention_proj_rows).to_csv(
        f"{sublayer_projection_dir}/attention_projections.csv", index=False
    )
    pd.DataFrame(mlp_proj_rows).to_csv(
        f"{sublayer_projection_dir}/mlp_projections.csv", index=False
    )

    # TODO: Should only be done on edit rows
    edit_metrics = [m for m in all_metrics if m.get("row_role") == "edit"]
    overall_res = calculate_averages(edit_metrics)

    all_metrics_payload = {
        "in_order": all_metrics,
        "skipped_rows": skipped,
    }

    base_name = f"AlphaEdit_{args.model_type}"
    with open(f"{args.metrics_save_dir}/{base_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics_payload, f, indent=4)
    with open(f"{args.metrics_save_dir}/{base_name}_res.json", "w", encoding="utf-8") as f:
        json.dump(overall_res, f, indent=4)

    print("Saved metrics and summaries.")


if __name__ == "__main__":
    main()
