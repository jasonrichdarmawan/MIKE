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


from analysis.patched_generation import get_hidden_states, generate_with_patching_all_layers
from analysis.utils import get_layer_names
def run_entity_description_e2_new(model, tokenizer, item, do_sample=False):
    # Replicate generate_entity_description_with_patching(source="e1") but use e2_new as the entity.
    # bridge = item["new_single_hops"][0]
    bridge = item["cloze_question"]
    source_prompt = item["new_single_hops"][1]["cloze"]
    # source_prompt =  item["cloze_question"]
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
        "e1_label": [bridge["answer"]],
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

    all_metrics = []
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
        is_e3_new_given = False
        if "post" in metrics and "accuracy" in metrics["post"]:
            if len(metrics["post"]["accuracy"]) > 1:
                is_e3_new_given = metrics["post"]["accuracy"][0] or metrics["post"]["accuracy"][1]
            elif len(metrics["post"]["accuracy"]) == 1:
                is_e3_new_given = metrics["post"]["accuracy"][0] 

        if is_e3_new_given:
            # sublayer projection
            metrics["project_sublayer"] = {}
            for target in ["attention", "mlp"]:
                try:
                    proj_result = run_project_sublayer(model, tokenizer, item, target)
                    metrics["project_sublayer"][f"{target}_layer"] = [_ for _ in proj_result["layer"]]
                    metrics["project_sublayer"][f"{target}_top_k_words"] = [_ for _ in proj_result["top_k_words"]]
                    metrics["project_sublayer"][f"{target}_top_k_probs"] = [_ for _ in proj_result["top_k_probs"]]
                    metrics["project_sublayer"][f"{target}_e2_ranks_per_layer"] = [_ for _ in proj_result["e2_ranks"]]
                    metrics["project_sublayer"][f"{target}_e3_ranks_per_layer"] = [_ for _ in proj_result["e3_ranks"]]
                    metrics["project_sublayer"][f"{target}_prediction_rank"] = [_ for _ in proj_result["prediction_rank"]]

                except Exception as e:
                    metrics["project_sublayer"][f"{target}_error"] = str(e)

            # entity description probing for e2_new decodability
            metrics["entity_description_e2_new"] = {}
            try:
                entity_result = run_entity_description_e2_new(model, tokenizer, item, do_sample=False)

                # find acceptable answers for e2_new
                bridge = item["new_single_hops"][0]
                e2_new_answers = (bridge.get("answer_alias", []) or []) + [bridge["answer"]]

                # will store (source_layer, target_layer) pairs where patching successfully generated e2_new
                successful_pairs = []
                
                for i, gen in enumerate(entity_result.get("generation", [])):
                    if gen and any(answer.lower() in str(gen).lower() for answer in e2_new_answers):
                        source_layer = entity_result["source_layer"][i]
                        target_layer = entity_result["target_layer"][i]
                        successful_pairs.append((source_layer, target_layer))

                metrics["entity_description_e2_new"] = {
                    "successful_patches": successful_pairs,
                    "count": len(successful_pairs),
                }
            except Exception as e:
                metrics["entity_description_e2_new"]["error"] = str(e)

            # knockout bridge representation
            metrics["knockout_bridge"] = {}
            #TODO: e1 last remove e2
            for knockout_source, knockout_target in [("last", "e1")]:
                try:
                    knockout_result = run_knockout_bridge(model, tokenizer, item, knockout_source, knockout_target, 
                                                         k=3, prompt="composition", default_decoding=False)
                    
                    # Check which layers, when knocked out, hurt e3_new generation
                    e3_new_answers = (item.get("new_answer_alias", []) or []) + [item["new_answer"]]
                    hurt_layers = []
                    for i, gen in enumerate(knockout_result.get("generation", [])):
                        # If generation doesn't contain e3_new answer, knockout hurt it
                        if gen and not any(answer.lower() in str(gen).lower() for answer in e3_new_answers):
                            layer = knockout_result["layer"][i]
                            hurt_layers.append(layer)
                    
                    metrics["knockout_bridge"][f"{knockout_source}_{knockout_target}"] = {
                        "hurt_layers": hurt_layers,
                        "count": len(hurt_layers),
                    }
                except Exception as e:
                    metrics["knockout_bridge"][f"{knockout_source}_{knockout_target}_error"] = str(e)
        
        all_metrics.append(metrics)

    os.makedirs(args.metrics_save_dir, exist_ok=True)

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
