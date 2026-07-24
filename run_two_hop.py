# %%

import argparse
import json
import os
import ast
from pathlib import Path

import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from datetime import datetime

from EasyEdit.easyeditor import (
    AlphaEditHyperParams,
    AlphaEditCircuitHyperParams,
    Revised_AlphaEditHyperParams,
)

from EasyEdit.easyeditor.util.alg_dict import ALG_DICT

from edit_utils import edit, evaluate, reevaluate_metrics_from_predictions
from test_cake import calculate_averages, save_checkpoint, load_checkpoint
from utils import setup_logger

from analysis.create_benchmark_cf import create_subset_cf
from analysis.two_hop_e2_json_create import convert_row_to_case
from analysis.project_sublayer import project_attention
from analysis.generate_entity_description import generate_entity_description_with_patching
from analysis.knockout_attention import generate_answer_with_attention_knockout
from analysis.utils import get_answers
from analysis.classify_results import load_entity_description_generations, load_attention_knockouts, load_sublayer_projections
from analysis.analyze_results import (
    load_layers, 
    get_sublayer_counts, 
    plot_stage_boxplots, 
    plot_min_entity_description_layers,
    load_entity_description,
    get_entity_counts,
    plot_layer_matrix,
)

# %%

# if __name__ == "__main__":
#     import os
#     import sys
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     sys.argv = [
#         "run_two_hop.py",
#         "--editing_method", "Revised_AlphaEdit",
#         "--model_type", "qwen3-4b-instruct",
#         # "--model_type", "tiny-aya-global",
#         # "--model_type", "Qwen/Qwen3-4B-Instruct-2507",
#         # "--model_type", "CohereLabs/tiny-aya-global",
#         "--dtype", "bfloat16",
#         "--sample_size", "4",
#         "--batch_size", "2",
#         # "--save_edited_model",
#         # "--load_edited_model",
#         # "--load_model_path", "checkpoints/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_20260430-143208/edit_num_0350_case_79415",
#         # "--load_eval_path", "output/CohereLabs/tiny-aya-global/two_hop_cf_e2/sequential_s1000_s32/Revised_AlphaEdit_20260430-145114",
#         "--do_edit",
#         "--save_edit_statistics",
#         "--eval_with_chat_template",
#         "--do_evaluate",
#         "--do_sublayer_projections",
#         "--do_patchscopes",
#         "--do_knockout_attention",
#     ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalaute in order: edit rows are edited, propagated rows are evaluated.")
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=["float32", "bfloat16"])
    parser.add_argument("--data_dir", default="./datasets", type=str)
    parser.add_argument("--swapping_strategy", default="e2", choices=["e2", "e3"], type=str)
    parser.add_argument("--sample_size", default=4, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--metrics_save_dir", default="./output", type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--save_edited_model", action="store_true")
    parser.add_argument("--load_edited_model", action="store_true")
    parser.add_argument("--load_model_path", default=None, type=str, help="Path to load the edited model checkpoint. Required if --load_edited_model is set.")
    parser.add_argument("--load_eval_path", default=None, type=str, help="Path to load evaluation results. Required if --load_edited_model is set.")
    parser.add_argument("--do_edit", action="store_true")
    parser.add_argument("--eval_with_chat_template", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true", help="Whether to do evaluation. By default, if --load_edited_model is set and existing metrics are found, evaluation will be skipped and existing metrics will be loaded.")
    parser.add_argument("--do_sublayer_projections", action="store_true", help="Whether to do sublayer projection. By default, if existing projection results are found, they will be loaded instead of re-running the projection.")
    parser.add_argument("--do_patchscopes", action="store_true", help="Whether to do patchscope computation even if results already exist.")
    parser.add_argument("--do_knockout_attention", action="store_true", help="Whether to do attention knockout even if results already exist.")
    parser.add_argument("--generation_batch_size", default=128, type=int)
    parser.add_argument(
        "--save_edit_statistics",
        action="store_true",
        help="Save target-loss, delta-norm, and update statistics as JSON.",
    )
    parser.add_argument(
        "--max_memory",
        type=lambda value: {
            int(device): memory
            for device, memory in (item.split(":", 1) for item in value.split(","))
        },
        help='Comma-separated GPU memory limits, e.g. "0:14GiB,1:18GiB".',
    )
    args = parser.parse_args()

    hparams = None
    if args.editing_method == "no_edit":
        pass
    elif args.editing_method in ["AlphaEdit", "AlphaEdit_1-rank", "AlphaEdit_1-rank_compute-matched"]:
        hparams = AlphaEditHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit/{args.model_type}.yaml')
        if args.editing_method == "AlphaEdit_1-rank_compute-matched":
            hparams.v_num_grad_steps = 125
    elif args.editing_method == 'CaKE': # Use PEFT
        pass
    elif args.editing_method in ["AlphaEdit_Circuit", "AlphaEdit_Circuit-no_layer"]:
        hparams = AlphaEditCircuitHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit_Circuit/{args.model_type}.yaml')
    elif args.editing_method == "Revised_AlphaEdit":
        hparams = Revised_AlphaEditHyperParams.from_hparams(f'./EasyEdit/hparams/Revised_AlphaEdit/{args.model_type}.yaml')
    else:
        raise ValueError(f"Unsupported editing method: {args.editing_method}")

    apply_algo = None
    if args.editing_method in [
        "AlphaEdit", "AlphaEdit_1-rank", "AlphaEdit_1-rank_compute-matched", 
        "AlphaEdit_Circuit", "AlphaEdit_Circuit-no_layer",
        "Revised_AlphaEdit",
    ]:
        alg_name = hparams.alg_name
        apply_algo = ALG_DICT[alg_name]

    model_name = hparams.model_name if hparams is not None else args.model_type
    
    if args.load_edited_model and args.load_model_path:
        datetime_str = args.load_model_path.split("/")[-2].split("_")[-1]
    elif args.load_eval_path:
        datetime_str = args.load_eval_path.split("/")[-1].split("_")[-1]
    else:
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    SAVE_METRICS_DIR = f"{args.metrics_save_dir}/{model_name}/two_hop_cf_{args.swapping_strategy}/sequential_s{args.sample_size}_s{args.batch_size}/{args.editing_method}_{datetime_str}"
    
    if args.save_edited_model:
        SAVE_CHECKPOINTS_DIR = f"{args.checkpoint_dir}/{model_name}/two_hop_cf_{args.swapping_strategy}/sequential_s{args.sample_size}_s{args.batch_size}/{args.editing_method}_{datetime_str}"
    
# %%

if __name__ == "__main__":
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    fh_path = f"{SAVE_METRICS_DIR}/output_{datetime_str}.log"
    os.makedirs(SAVE_METRICS_DIR, exist_ok=True)
    logger = setup_logger(log_file=fh_path)
    print(f"Saving output log to {fh_path}")
    
# %%

def assign_edit_ids(data):
    edit_id = 1
    for case in data:
        if case["row_role"] == "edit":
            for rewrite in case["requested_rewrite"]:
                rewrite["case_id"] = case["case_id"]
                rewrite["edit_id"] = edit_id
                edit_id += 1
    return data

def _prepare_cases_for_editing(file_path: Path):
    """
    file_path: "datasets/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2.csv"
    Convert CSV rows to JSON cases
    """
    df = pd.read_csv(file_path)

    # based on analysis:create_benchmark_cf.py 120:125
    # remove annoying column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.set_index("id")
    df["propagated_from"] = df.apply(lambda row:
        ast.literal_eval(row["propagated_from"])
        if row["row_role"] == "propagated_eval"
        else row["propagated_from"], axis=1
    )
    df["swapped_with"] = df["swapped_with"].astype("Int64")

    swapping_strategy = file_path.stem.split("_")[-1]  # e.g., "e2" or "e3"
    stats, subset_df = create_subset_cf(df=df, swapping_strategy=swapping_strategy)

    # save the benchmark subset CSV for reference
    benchmark_csv_path = Path(str(file_path).replace("two_hop_cf", "two_hop_cf_benchmark"))
    benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)

    subset_df_out = subset_df.reset_index()
    subset_df_out.to_csv(benchmark_csv_path, index=False)

    # create json file
    cases = [convert_row_to_case(row) for _, row in subset_df_out.iterrows()]
    cases = assign_edit_ids(cases)

    output_json_path = benchmark_csv_path.with_name(f"{benchmark_csv_path.stem}-clke.json")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    stats.update({
        "benchmark_csv_path": str(benchmark_csv_path),
        "benchmark_json_path": str(output_json_path),
    })

    return stats, subset_df, cases

if __name__ == "__main__":
    DATA_DIR = Path(args.data_dir) / model_name
    FILE_PATH = DATA_DIR / f"two_hop_cf_{args.swapping_strategy}.csv"
    
    stats, cf_benchmark, cases = _prepare_cases_for_editing(
        file_path=FILE_PATH,
    )
    stats_str = " | ".join([f"{k}: {v}" for k, v in stats.items()])
    print(f"Benchmark preparation completed | {stats_str}")

# %%

if __name__ == "__main__":
    if args.load_edited_model and args.load_model_path:
        model, tokenizer = load_checkpoint(
            model_name=model_name,
            device_map="cuda",
            dtype=args.dtype,
            editing_method=args.editing_method,
            load_path=args.load_model_path,
        )
        print(f"Loaded edited model by {args.editing_method} from checkpoint {args.load_model_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=args.max_memory if args.max_memory else None,
            dtype=getattr(torch, args.dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
# %%

if __name__ == "__main__":
    cf_benchmark = cf_benchmark[:args.sample_size]
    cases = cases[:args.sample_size]
    n_edit = sum(1 for x in cases if x["row_role"] == "edit")
    n_propagated = sum(1 for x in cases if x["row_role"] == "propagated_eval")
    print(f"Loaded {len(cases)} rows | edit={n_edit} propagated_eval={n_propagated}")
    
# %%

if __name__ == "__main__":
    if args.do_edit and not (args.load_edited_model and args.load_model_path):
        edit_statistics = []

        edit_cases = [item for item in cases if item["row_role"] == "edit"]
        # split edit_cases into list of lists, each of length args.batch_size (except possibly the last one)
        edit_cases = [edit_cases[i:i + args.batch_size] for i in range(0, len(edit_cases), args.batch_size)]
        if args.editing_method == "no_edit":
            pass
        elif args.editing_method in [
            "AlphaEdit", "AlphaEdit_1-rank", "AlphaEdit_1-rank_compute-matched",
            "AlphaEdit_Circuit", "AlphaEdit_Circuit-no_layer",
            "Revised_AlphaEdit",
        ]:
            collect_statistics = (
                args.save_edit_statistics
                and hparams.alg_name in [
                    "AlphaEdit", "AlphaEdit_Circuit", "Revised_AlphaEdit"
                ]
            )
            for batch in tqdm(edit_cases, desc="Editing the model"):
                batch_statistics = []

                edit_item_groups = [[item] for item in batch] if args.editing_method in ["AlphaEdit_1-rank", "AlphaEdit_1-rank_compute-matched"] else [batch]
                for edit_items in edit_item_groups:
                    edit_result = edit(
                        model=model,
                        tokenizer=tokenizer,
                        items=edit_items,
                        hparams=hparams,
                        apply_algo=apply_algo,
                        return_statistics=collect_statistics,
                    )
                    if collect_statistics:
                        model, request_statistics = edit_result
                        batch_statistics.extend(request_statistics)
                    else:
                        model = edit_result
                
                if collect_statistics:
                    for statistic in batch_statistics:
                        statistic["edit_batch_start"] = batch[0]["case_id"]
                        statistic["edit_batch_end"] = batch[-1]["case_id"]
                        statistic["editing_method"] = args.editing_method
                    edit_statistics.extend(batch_statistics)
                    statistics_path = os.path.join(
                        SAVE_METRICS_DIR, f"edit_statistics.json"
                    )
                    with open(statistics_path, "w", encoding="utf-8") as f:
                        json.dump(edit_statistics, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported editing method in edit loop: {args.editing_method}")
        
# %%

if __name__ == "__main__":
    if args.save_edited_model and not args.load_edited_model:
        checkpoint_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            editing_method=args.editing_method,
            checkpoint_root=SAVE_CHECKPOINTS_DIR,
            edit_num=n_edit,
            case_id=edit_cases[-1][-1]["case_id"],
        )
        print(f"Edited model checkpoint saved to {checkpoint_path}")      

# %%

def do_evaluate(
    model, 
    tokenizer, 
    cases, 
    generation_batch_size, 
    editing_method, 
    eval_with_chat_template,
    save_metrics_dir,
    **kwargs,
):
    all_metrics = {}
    
    eval_cases = [cases[i:i + generation_batch_size] for i in range(0, len(cases), generation_batch_size)]
    for idx, batch in enumerate(tqdm(eval_cases, desc="Evaluating the model")):
        if editing_method in [
            "no_edit", 
            "AlphaEdit", "AlphaEdit_1-rank", "AlphaEdit_1-rank_compute-matched",
            "AlphaEdit_Circuit", "AlphaEdit_Circuit-no_layer",
            "Revised_AlphaEdit",
        ]:
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                items=batch,
                alg_name=editing_method,
                eval_with_chat_template=eval_with_chat_template,
                generation_batch_size=generation_batch_size,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported editing method in evaluation loop: {editing_method}")
        
        for key in metrics.keys():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].extend(metrics[key])
            
            folder_path = f"{save_metrics_dir}/{key}"
            os.makedirs(folder_path, exist_ok=True)
            
            metrics_str = "metrics.json"
            file_path = os.path.join(folder_path, metrics_str)
            print(f"Saving {key} metrics to {file_path}")
            json.dump(all_metrics[key], open(file_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            
            res = calculate_averages(data=all_metrics[key])
            print(f"Batch {idx} | key={key} averages:\n{json.dumps(res, indent=4, ensure_ascii=False)}")
            res_str = "res.json"
            res_filepath = os.path.join(folder_path, res_str)
            json.dump(res, open(res_filepath, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    
        if idx % 2 == 0:
            for key in metrics.keys():
                print(f"Batch {idx} | key={key}:\n{json.dumps(metrics[key][0], indent=4, ensure_ascii=False)}")
    
    return all_metrics

if __name__ == "__main__":
    all_metrics = {}
    
    kwargs = {}
    kwargs["languages"] = ["en"]
    if not args.do_evaluate:
        for language in kwargs["languages"]:
            metrics_path = os.path.join(SAVE_METRICS_DIR, language, "metrics.json")
            with open(metrics_path, "r", encoding="utf-8") as f:
                all_metrics[language] = json.load(f)
            print(f"Loaded existing metrics from {metrics_path}")
            
        extract_answer = "answer_tags" if args.eval_with_chat_template else False
        
        all_metrics = reevaluate_metrics_from_predictions(
            items=cases,
            all_metrics=all_metrics,
            languages=kwargs["languages"],
            extract_answer=extract_answer,
        )
        
        for key in all_metrics.keys():
            metrics_str = "metrics.json"
            metrics_filepath = f'{args.load_eval_path}/{key}/{metrics_str}'
            print(f"Saving reevaluated metrics to {metrics_filepath}")
            json.dump(all_metrics[key], open(metrics_filepath, 'w'), indent=4, ensure_ascii=False)
            
            res = calculate_averages(all_metrics[key])
            print(f"Results {key}:\n{json.dumps(res, indent=4)}")
            
            folder_path = f'{args.load_eval_path}/{key}'
            res_str = "res.json"
            res_filepath = f'{folder_path}/{res_str}'
            print(f"Saving reevaluated results to {res_filepath}")
            json.dump(res, open(res_filepath, 'w'), indent=4)
    else:
        all_metrics = do_evaluate(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            generation_batch_size=args.generation_batch_size,
            editing_method=args.editing_method,
            eval_with_chat_template=args.eval_with_chat_template,
            save_metrics_dir=SAVE_METRICS_DIR,
            **kwargs
        )

# %%

def update_cf_benchmark(cf_benchmark, cases, all_metrics):
    if len(cf_benchmark) != len(cases) or len(cf_benchmark) != len(all_metrics["en"]):
        raise ValueError(f"Length mismatch: cf_benchmark={len(cf_benchmark)}, cases={len(cases)}, metrics={len(all_metrics['en'])}")
    
    def _get_new(row, key, default):
        val = row[key]
        if pd.isna(val):
            return default
        return val
    
    for (index, row), case, metric in zip(cf_benchmark.iterrows(), cases, all_metrics["en"]):
        if index != case["case_id"] or index != metric["case_id"]:
            raise ValueError(f"Index mismatch: cf_benchmark={index} vs cases={case['case_id']} vs metrics={metric['case_id']}")
        cf_benchmark.loc[index, "e2"] = _get_new(row, "e2_new", default=row["e2"])
        cf_benchmark.loc[index, "e3"] = _get_new(row, "e3_new", default=row["e3"])
        cf_benchmark.loc[index, "e2_label"] = _get_new(row, "e2_label_new", default=row["e2_label"])
        cf_benchmark.loc[index, "e3_label"] = _get_new(row, "e3_label_new", default=row["e3_label"])
        cf_benchmark.loc[index, "e2_aliases"] = _get_new(row, "e2_aliases_new", default=row["e2_aliases"])
        cf_benchmark.loc[index, "e3_aliases"] = _get_new(row, "e3_aliases_new", default=row["e3_aliases"])
        cf_benchmark.loc[index, "first_hop_correct"] = metric["post"]["hop_wise"][0][0]
        cf_benchmark.loc[index, "second_hop_correct"] = metric["post"]["hop_wise"][1][0]
        cf_benchmark.loc[index, "composition_correct"] = metric["post"]["accuracy"][0]

if __name__ == "__main__":
    update_cf_benchmark(cf_benchmark, cases, all_metrics)
    cf_benchmark_output_path = f"{SAVE_METRICS_DIR}/two_hop_cf_benchmark_{args.swapping_strategy}.csv"
    cf_benchmark.to_csv(cf_benchmark_output_path)
    print(f"Updated cf_benchmark saved to {cf_benchmark_output_path}")

# %%

if __name__ == "__main__":
    composition_correct_mask = (
        (cf_benchmark["first_hop_correct"] == True) 
        & (cf_benchmark["second_hop_correct"] == True) 
        & (cf_benchmark["composition_correct"] == True)
    )
    composition_correct_subset = cf_benchmark[composition_correct_mask]
    os.makedirs(f"{SAVE_METRICS_DIR}/composition_correct", exist_ok=True)
    csv_path = f"{SAVE_METRICS_DIR}/composition_correct/two_hop_cf_benchmark_{args.swapping_strategy}.csv"
    composition_correct_subset.to_csv(csv_path)
    print(f"Composition correct subset saved to {csv_path}")

    composition_incorrect_mask = (
        (cf_benchmark["first_hop_correct"] == True)
        & (cf_benchmark["second_hop_correct"] == True)
        & (cf_benchmark["composition_correct"] == False)
    )
    composition_incorrect_subset = cf_benchmark[composition_incorrect_mask]
    os.makedirs(f"{SAVE_METRICS_DIR}/composition_incorrect", exist_ok=True)
    csv_path = f"{SAVE_METRICS_DIR}/composition_incorrect/two_hop_cf_benchmark_{args.swapping_strategy}.csv"
    composition_incorrect_subset.to_csv(csv_path)
    print(f"Composition incorrect subset saved to {csv_path}")

    subsets = {
        "composition_correct": composition_correct_subset,
        "composition_incorrect": composition_incorrect_subset,
    }
    subset_sizes = {name: len(df) for name, df in subsets.items()}
    print(
        "Subset sizes | " +
        " ".join([f"{name}={size}" for name, size in subset_sizes.items()])
    )

# %%

def create_empty_subset_artifacts(
    save_metrics_dir: str,
    subset_str: str,
):
    base_dir = Path(save_metrics_dir) / subset_str
    os.makedirs(base_dir, exist_ok=True)

    empty_files = {
        base_dir / "sublayer_projection" / "attention_projections.csv": [
            "id", "layer", "top_k_words", "top_k_probs", "e2_ranks", "e3_ranks", "prediction_rank"
        ],
        base_dir / "sublayer_projection" / "mlp_projections.csv": [
            "id", "layer", "top_k_words", "top_k_probs", "e2_ranks", "e3_ranks", "prediction_rank"
        ],
        base_dir / "entity_description" / "e1_entity_description_s0.csv": [
            "id", "source_layer", "target_layer", "generation"
        ],
        base_dir / "entity_description" / "last_entity_description_s0.csv": [
            "id", "source_layer", "target_layer", "generation"
        ],
        base_dir / "attention_knockout" / "last_e1_attention_knockout_k3_composition.csv": [
            "id", "layer", "generation"
        ],
    }

    for path, columns in empty_files.items():
        os.makedirs(path.parent, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path)
    print(f"Created empty placeholder artifacts for {subset_str} at {base_dir}")

if __name__ == "__main__":
    for subset_str, subset in subsets.items():
        if subset.empty:
            create_empty_subset_artifacts(
                save_metrics_dir=SAVE_METRICS_DIR,
                subset_str=subset_str,
            )

# %%

def do_sublayer_projections(
    subset: pd.DataFrame,
    generation_batch_size: int,
    save_metrics_dir: str,
    subset_str: str = "composition_correct",
):
    """
    Code is following the same pattern as `analysis/project_sublayer.py`
    """
    df = subset.copy(deep=True)
    records = df.reset_index().rename(columns={"index": "id"}).to_dict(orient="records")
    for row in records:
        row.update(get_answers(row, key="e3"))
        row.update(get_answers(row, key="e2"))
    
    for target in ["attention", "mlp"]:
        generation_frames = []
        for start in tqdm(range(0, len(records), generation_batch_size), desc=f"Generating {target} projections"):
            batch = records[start:start + generation_batch_size]
            entries = {k: [row[k] for row in batch] for k in batch[0].keys()}
            generated = project_attention(
                entries=entries,
                model=model,
                tokenizer=tokenizer,
                target=target,
            )
            generation_frames.append(pd.DataFrame(generated))
        generations = pd.concat(generation_frames)
        generations = generations.sort_values(["id", "layer"])
        generations = generations.reset_index(drop=True)
        generations["top_k_words"] = generations["top_k_words"].apply(list)
        generations["top_k_probs"] = generations["top_k_probs"].apply(list)
        generations["e2_ranks"] = generations["e2_ranks"].apply(list)
        generations["e3_ranks"] = generations["e3_ranks"].apply(list)
        output_path = f"{save_metrics_dir}/{subset_str}/sublayer_projection/{target}_projections.csv"
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        generations.to_csv(output_path, index=False)
        print(f"Sublayer projection for {target} saved to {output_path}")


if __name__ == "__main__":

    if args.do_sublayer_projections:
        for subset_str, subset in subsets.items():
            if subset.empty:
                print(f"Skipping sublayer projection for {subset_str}: empty subset")
                continue
            do_sublayer_projections(
                subset=subset,
                generation_batch_size=args.generation_batch_size,
                save_metrics_dir=SAVE_METRICS_DIR,
                subset_str=subset_str,
            )

# %%

def do_patchscopes(
    subset: pd.DataFrame,
    generation_batch_size: int,
    save_metrics_dir: str,
    subset_str: str = "composition_correct",
):
    """
    Note: Code is following the same pattern as `analysis/generate_entity_description.py`
    """
    df = subset.copy(deep=True)
    target_prompt = (
        "Syria: Syria is a country in the Middle East, " +
        "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, " +
        "Samsung: Samsung is a South Korean multinational corporation, " +
        "x"
    )
    df["target_prompt"] = target_prompt
    records = df.reset_index().rename(columns={"index": "id"}).to_dict(orient="records")
    for source in ["e1", "last"]:
        generation_frames = []
        for start in tqdm(range(0, len(records), generation_batch_size), desc=f"Generating entity description with source {source}"):
            batch = records[start:start + generation_batch_size]
            entries = {k: [row[k] for row in batch] for k in batch[0].keys()}
            generated = generate_entity_description_with_patching(
                entries=entries,
                model=model,
                tokenizer=tokenizer,
                source=source,
                do_sample=False,
            )
            generation_frames.append(pd.DataFrame(generated))
        generations = pd.concat(generation_frames)
        generations = generations.sort_values(["id", "source_layer", "target_layer"])
        generations = generations.reset_index(drop=True)
        output_path = f"{save_metrics_dir}/{subset_str}/entity_description/{source}_entity_description_s0.csv"
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        generations.to_csv(output_path)
        print(f"Entity description for {source} saved to {output_path}")

if __name__ == "__main__":
    if args.do_patchscopes:
        for subset_str, subset in subsets.items():
            if subset.empty:
                print(f"Skipping patchscope generation for {subset_str}: empty subset")
                continue
            do_patchscopes(
                subset=subset,
                generation_batch_size=args.generation_batch_size,
                save_metrics_dir=SAVE_METRICS_DIR,
                subset_str=subset_str,
            )

# %%

def do_knockout_attention(
    subset: pd.DataFrame,
    generation_batch_size: int,
    save_metrics_dir: str,
    subset_str: str = "composition_correct",
):
    """
    Note: Code is following the same pattern as `analysis/knockout_attention.py`
    """
    df = subset.copy(deep=True)
    records = df.reset_index().rename(columns={"index": "id"}).to_dict(orient="records")
    knockout_source = "last"
    knockout_target = "e1"
    k = 3
    prompt = "composition"
    generation_frames = []
    for start in tqdm(range(0, len(records), generation_batch_size), desc="Generating attention knockout results"):
        batch = records[start:start + generation_batch_size]
        entries = {k: [row[k] for row in batch] for k in batch[0].keys()}
        generated = generate_answer_with_attention_knockout(
            entries=entries,
            model=model,
            tokenizer=tokenizer,
            default_decoding=False,
            knockout_source=knockout_source,
            knockout_target=knockout_target,
            k=k,
            prompt="composition",
        )
        generation_frames.append(pd.DataFrame(generated))
    generations = pd.concat(generation_frames)
    generations = generations.sort_values(["id", "layer"])
    generations = generations.reset_index(drop=True)
    output_path = (
        f"{save_metrics_dir}/{subset_str}/attention_knockout/"
        f"{knockout_source}_{knockout_target}_attention_knockout_k{k}_{prompt}.csv"
    )
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    generations.to_csv(output_path)
    print(f"Attention knockout results saved to {output_path}")

if __name__ == "__main__":
    if args.do_knockout_attention:
        for subset_str, subset in subsets.items():
            if subset.empty:
                print(f"Skipping attention knockout for {subset_str}: empty subset")
                continue
            do_knockout_attention(
                subset=subset,
                generation_batch_size=args.generation_batch_size,
                save_metrics_dir=SAVE_METRICS_DIR,
                subset_str=subset_str,
            )

# %%

def do_classify_results(
    results_dir: str,
    swapping_strategy: str,
):
    """
    Note: Code is following the same pattern as `analysis/classify_results.py`
    """
    entity_description_layers = load_entity_description_generations(dataset_dir=results_dir)
    attention_knockout_layers = load_attention_knockouts(dataset_dir=results_dir)
    sublayer_projection_layers = load_sublayer_projections(dataset_dir=results_dir)
    
    two_hop = pd.read_csv(f"{results_dir}/two_hop_cf_benchmark_{swapping_strategy}.csv")
    base = two_hop[["id", "composition_correct"]]
    df = base \
        .merge(entity_description_layers["e1"]["e2"], on="id", how="left") \
        .merge(entity_description_layers["e1"]["e3"], on="id", how="left") \
        .merge(entity_description_layers["last"]["e2"], on="id", how="left") \
        .merge(entity_description_layers["last"]["e3"], on="id", how="left") \
        .merge(sublayer_projection_layers["attention"]["prediction"], on="id", how="left") \
        .merge(sublayer_projection_layers["attention"]["e2"], on="id", how="left") \
        .merge(sublayer_projection_layers["mlp"]["prediction"], on="id", how="left") \
        .merge(sublayer_projection_layers["mlp"]["e2"], on="id", how="left") \
        .merge(attention_knockout_layers["last"]["e1"], on="id", how="left")
    
    df.to_csv(f"{results_dir}/layers.csv")

if __name__ == "__main__":
    for subset_str, subset in subsets.items():
        do_classify_results(
            results_dir=f"{SAVE_METRICS_DIR}/{subset_str}",
            swapping_strategy=args.swapping_strategy,
        )
    
# %%

def do_analyze_results(
    save_metrics_dir: str,
    n_layers: int,
):
    """
    Note: Code is following the same pattern as `analysis/analyze_results.py`
    """
    layers = {
        "correct": load_layers(path=f"{save_metrics_dir}/composition_correct/layers.csv"),
        "incorrect": load_layers(path=f"{save_metrics_dir}/composition_incorrect/layers.csv"),
    }
    print(f"Correct setting example count: {len(layers['correct'])}")
    print(f"Incorrect setting example count: {len(layers['incorrect'])}")
    print(
        f"Correct setting prediction in sublayers:\n", 
        f"{get_sublayer_counts(df=layers['correct'], target='prediction')}"
    )
    print(
        f"Incorrect setting prediction in sublayers:\n",
        f"{get_sublayer_counts(df=layers['incorrect'], target='prediction')}"
    )

    plots_dir = f"{save_metrics_dir}/plots"
    plot_stage_boxplots(
        correct_df=layers["correct"],
        incorrect_df=layers["incorrect"],
        path=plots_dir,
    )
    print(f"Stage boxplots saved to {plots_dir}")
    
    path = f"{plots_dir}/min_entity_description_layers.pdf"
    plot_min_entity_description_layers(
        df=pd.concat([layers['correct'], layers['incorrect']]),
        path=path,
        layers=n_layers,
    )
    print(f"Min entity description layer plots saved to {path}")
    
    entity_description_dir = f"{save_metrics_dir}/composition_correct/entity_description"
    entity_descriptions = {
        "e1": load_entity_description(
            path=entity_description_dir,
            source="e1",
            seeds=1,
        ),
        "last": load_entity_description(
            path=entity_description_dir,
            source="last",
            seeds=1,
        ),
    }

    print(
        f"Correct setting entity description counts:\n",
        get_entity_counts(df=layers["correct"])
    )
    print(
        f"Incorrect setting entity description counts:\n",
        get_entity_counts(df=layers["incorrect"])
    )
    
    if not entity_descriptions["e1"].empty:
        path = f"{plots_dir}/entity_description_e1_e2.pdf"
        plot_layer_matrix(
            generations=entity_descriptions["e1"],
            target_classification="e2",
            path=path,
        )
        print(f"Entity description layer matrix plots saved to {path}")
    if not entity_descriptions["last"].empty:
        path = f"{plots_dir}/entity_description_last_e2.pdf"
        plot_layer_matrix(
            generations=entity_descriptions["last"],
            target_classification="e2",
            path=f"{plots_dir}/entity_description_last_e2.pdf",
        )
        print(f"Entity description layer matrix plots saved to {path}")
        
        path = f"{plots_dir}/entity_description_last_e3.pdf"
        plot_layer_matrix(
            generations=entity_descriptions["last"],
            target_classification="e3",
            path=path,
        )
        print(f"Entity description layer matrix plots saved to {path}")

if __name__ == "__main__":
    n_layers = None
    if model_name in [
        "Qwen/Qwen3-4B-Instruct-2507",
        "CohereLabs/tiny-aya-global",
        "Qwen/Qwen2.5-7B-Instruct",
        "CohereLabs/aya-expanse-8b",
    ]:
        n_layers = model.config.num_hidden_layers
    else:
        raise ValueError(f"Unsupported model for layer number: {model_name}")
    do_analyze_results(
        save_metrics_dir=SAVE_METRICS_DIR,
        n_layers=n_layers,
    )

# %%