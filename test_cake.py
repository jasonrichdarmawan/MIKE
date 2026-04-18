# %%

from EasyEdit.easyeditor import (
    LoRAHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    R_ROMEHyperParams,
    AlphaEditHyperParams,
    WISEHyperParams,
    WISE,
    AlphaEditCircuitHyperParams,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm.auto import tqdm
from edit_utils import (
    edit,
    evaluate_mello,
    get_sentence_embeddings,
    edit_wise,
    edit_ifmet,
    edit_cake,
    evaluate,
)
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import os
import logging
from utils import StreamHandler, FileHandler, BufferingLoggerWriter
import sys
from datetime import datetime

from glue_eval.glue_eval import GLUEEval

# %%

def load_checkpoint(model_name, device_map, dtype, editing_method, load_path, **kwargs):
    """
    Caveat: Only guarantee loaded checkpoint works for evaluation, not continuing editing.
    """
    model = None
    tokenizer = None
    
    if editing_method in ["LoRA", "ROME", "R-ROME", "MEMIT", "AlphaEdit", "AlphaEdit_Circuit", "IFMET", "CaKE"]:
        model = AutoModelForCausalLM.from_pretrained(load_path, device_map=device_map, dtype=getattr(torch, dtype))
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    elif editing_method == "WISE":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, dtype=getattr(torch, dtype))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        editor = WISE(model=model, config=kwargs["hparams"], device=model.device)
        editor.load(load_path=load_path)
    else:
        raise NotImplementedError(f"Editing method {editing_method} not implemented for loading checkpoint.")
        
    return model, tokenizer

def save_checkpoint(model, tokenizer, editing_method, checkpoint_root, edit_num, case_id, **kwargs):
    checkpoint_name = f"edit_num_{edit_num:04d}_case_{case_id:04d}"
    checkpoint_path = os.path.join(checkpoint_root, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    if editing_method in ["LoRA", "ROME", "R-ROME", "MEMIT", "AlphaEdit", "AlphaEdit_Circuit", "IFMET", "CaKE"]:
        model.save_pretrained(
            checkpoint_path,
            safe_serialization=True,
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(checkpoint_path)
    elif editing_method in ["WISE"]:
        editor = WISE(model=model, config=kwargs["hparams"], device=model.device)
        editor.save(f"{checkpoint_path}/wise.pt")
    else:
         raise NotImplementedError(f"Editing method {editing_method} not implemented for saving checkpoint.")

    metadata = {
        "edit_num": edit_num,
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
    }
    with open(os.path.join(checkpoint_path, "checkpoint_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return checkpoint_path

def calculate_averages(data):
    """
    Caveat:
    - CaKE's implementation only evaluate the 1st multi-hop question
    
    - "overall_hop_wise_average": the average of correctly answering the single-hop questions.
    - "accuracy_averages": the average of correctly answering the multi-hop questions.
    """
    total_cases = len(data)
    accuracy_case_averages = []
    hop_wise_case_averages = [] 
    
    for case in data:
        hop_wise = case['post']['hop_wise']
        accuracy = case['post']['accuracy']
        case_hop_true_count = 0
        total_hops = 0
        for hop_pair in hop_wise:
            case_hop_true_count += sum(hop_pair)
            total_hops += len(hop_pair)
        
        case_hop_average = case_hop_true_count / total_hops
        hop_wise_case_averages.append(case_hop_average)
        
        accuracy_true_count = sum(accuracy)
        accuracy_average = accuracy_true_count / len(accuracy)
        accuracy_case_averages.append(accuracy_average)
    
    overall_hop_wise_avg = sum(hop_wise_case_averages) / total_cases
    accuracy_avg = sum(accuracy_case_averages) / total_cases
    
    return {
        "total_cases": total_cases,
        "overall_hop_wise_average": f"{overall_hop_wise_avg:.3f}",
        "accuracy_averages": f"{accuracy_avg:.3f}"
    }

# %%

# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     sys.argv = ["", 
#         "--editing_method", "ROME",
#         # "--model_type", "Qwen/Qwen3-4B-Instruct-2507",
#         # "--model_type", "CohereLabs/tiny-aya-global",
#         # "--model_type", "qwen3-4b-instruct",
#         # "--model_type", "llama3-8b",
#         "--model_type", "tiny-aya-global",
#         "--dtype", "bfloat16",
#         # "--eval_with_chat_template",
#         "--batch_size", "2",
#         "--generation_batch_size", "16",
#         "--sample_size", "4",
#         "--data_name", "MQuAKE-CF-3k-v2-clke",
#         "--glue_eval_steps", "0",
#         "--glue_eval_size", "2",
#         "--main_eval_steps", "0",
#         "--save_edited_model",
#         "--load_edited_model",
#         "--load_path", "./checkpoints/CohereLabs/tiny-aya-global/MQuAKE-CF-3k-v2-clke/sequential0_s4_b2/ROME/edit_num_0004_case_0004",
#     ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--data_name', default=None, type=str, choices=["MQuAKE-CF-3k-v2-clke"])
    parser.add_argument('--model_type', default=None,type=str)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--eval_with_chat_template', action='store_true')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--generation_batch_size', default=16, type=int)
    parser.add_argument("--sample_size", default=2999, type=int)
    parser.add_argument("--glue_eval_steps", default=5, type=int)
    parser.add_argument("--glue_eval_size", default=100, type=int)
    parser.add_argument('--main_eval_steps', default=0, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
    parser.add_argument('--save_edited_model', action='store_true')
    parser.add_argument('--load_edited_model', action='store_true')
    parser.add_argument('--load_path', default=None, type=str)
    args = parser.parse_args()

    hparams = None
    if args.editing_method in ['no_edit', "MeLLo", "MeLLo_KV"]: # In-Context Learning Editing Methods
        pass
    elif args.editing_method == 'LoRA':
        hparams = LoRAHyperParams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
    elif args.editing_method == 'ROME':
        hparams = ROMEHyperParams.from_hparams(f'./EasyEdit/hparams/ROME/{args.model_type}.yaml')
    elif args.editing_method == 'MEMIT':
        hparams = MEMITHyperParams.from_hparams(f'./EasyEdit/hparams/MEMIT/{args.model_type}.yaml')
    elif args.editing_method == "R-ROME":
        hparams = R_ROMEHyperParams.from_hparams(f'./EasyEdit/hparams/R-ROME/{args.model_type}.yaml')
    elif args.editing_method == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit/{args.model_type}.yaml')
    elif args.editing_method == "WISE":
        hparams = WISEHyperParams.from_hparams(f'./EasyEdit/hparams/WISE/{args.model_type}.yaml')
    elif args.editing_method == "IFMET":
        hparams = MEMITHyperParams.from_hparams(f'./EasyEdit/hparams/IFMET/{args.model_type}-shallow.yaml')
        hparams_deeper = MEMITHyperParams.from_hparams(f'./EasyEdit/hparams/IFMET/{args.model_type}-deeper.yaml')
    elif args.editing_method == 'CaKE': # Use PEFT
        pass
    elif args.editing_method == "AlphaEdit_Circuit":
        hparams = AlphaEditCircuitHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit_Circuit/{args.model_type}.yaml')
    else:
        raise ValueError(f"Unsupported editing method: {args.editing_method}")
    
    # In-Context Learning Editing Methods (e.g., MeLLo) do not require apply_algo
    apply_algo = None
    if args.editing_method in ["LoRA", "ROME", "MEMIT", "R-ROME", "AlphaEdit", "WISE", "AlphaEdit_Circuit"]:
        apply_algo = ALG_DICT[hparams.alg_name]
    elif args.editing_method == "IFMET":
        apply_algo = ALG_DICT["MEMIT"]

    model_name = (
        hparams.model_name
        if hasattr(hparams, 'model_name') and hparams.model_name
        else args.model_type
    )
    sequential_str = f"sequential{args.main_eval_steps}"
    SAVE_METRICS_DIR = f'{args.metrics_save_dir}/{model_name}/{args.data_name}/{sequential_str}_s{args.sample_size}_b{args.batch_size}/{args.editing_method}'
    os.makedirs(SAVE_METRICS_DIR, exist_ok=True)
    
    if args.save_edited_model:
        SAVE_CHECKPOINTS_DIR = f'{args.checkpoint_dir}/{model_name}/{args.data_name}/{sequential_str}_s{args.sample_size}_b{args.batch_size}/{args.editing_method}'
        os.makedirs(SAVE_CHECKPOINTS_DIR, exist_ok=True)
        
# %%

if __name__ == "__main__":
    os.makedirs(SAVE_METRICS_DIR, exist_ok=True)
    
    fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)-9s %(message)s')
    
    sh = StreamHandler()
    sh.setFormatter(fmt)
    
    fh_str = "output.log" if not args.load_edited_model else f"output_loaded.log"
    fh_path = f"{SAVE_METRICS_DIR}/{fh_str}"
    fh = FileHandler(fh_path, mode='w')
    fh.setFormatter(fmt)
    
    logger = logging.getLogger("CaKE_Test")
    logger.propagate = False  # Prevent log messages from being propagated to the root logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    # Redirect stdout to the logger
    sys.stdout = BufferingLoggerWriter(logger, logging.INFO)
    sys.stderr = BufferingLoggerWriter(logger, logging.WARNING)
    print(f"Saving output to {fh_path}")
    
# %%

if __name__ == "__main__":
    if args.load_edited_model:
        kwargs = {}
        if args.editing_method == "WISE":
            kwargs["hparams"] = hparams
        model, tokenizer = load_checkpoint(
            model_name=model_name, 
            device_map="cuda", 
            dtype=args.dtype, 
            editing_method=args.editing_method, 
            load_path=args.load_path,
            **kwargs
        )
        print(f"Loaded edited model by {args.editing_method} from checkpoint {args.load_path}.")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", dtype=getattr(torch, args.dtype))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# %%

if __name__ == "__main__":
    data = json.load(open(f'./datasets/{args.data_name}.json','r'))
    if args.sample_size > len(data):
        raise ValueError(f"Sample size {args.sample_size} exceeds dataset size {len(data)}.")
    data = data[:args.sample_size]
    
# %%

if __name__ == "__main__":
    if args.editing_method in ['MeLLo', 'MeLLo_KV']:
        contriever_model_name = "facebook/contriever-msmarco"
        contriever = AutoModel.from_pretrained(contriever_model_name, device_map="cuda")
        contriever_tokenizer = AutoTokenizer.from_pretrained(contriever_model_name)
        with open('./prompts/MeLLo-prompt.txt', 'r') as f:
            task_prompt = f.read()
        new_facts = set()
        for d in data:
            for r in d["requested_rewrite"]:
                new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
        new_facts = list(new_facts)
        embeddings = get_sentence_embeddings(
            contriever=contriever,
            contriever_tokenizer=contriever_tokenizer,
            sentences=new_facts,
        )

# %%

def do_glue_eval(save_dir, edit_num, case_ids, number_of_tests):
    glue_results = {
        "edit_num": edit_num,
        "case_ids": case_ids,
    }
    
    glue_eval_path = f'{save_dir}/glue_eval'
    os.makedirs(glue_eval_path, exist_ok=True)
    out_file = (
        f'{glue_eval_path}/case_{{case:04d}}.json'.format(case=case_ids[-1])
        if case_ids
        else f'{glue_eval_path}/base.json'
    )
    tokenizer.padding_side = "left"
    glue_eval = GLUEEval(
        model,
        tokenizer,
        number_of_tests=number_of_tests,
    )
    glue_results = glue_eval.evaluate(glue_results,
        out_file,
        nli_flag=True,
        sst_flag=True,
        cola_flag=True,
        rte_flag=True,
        mmlu_flag=True,
        mrpc_flag=True,
    )
    output_filename = out_file.replace('.json', '_glue.json')
    with open(output_filename, 'w') as f:
        json.dump(glue_results, f, indent=4)
    
    return output_filename

def do_evaluation(
    all_metrics,
    model, 
    tokenizer, 
    items, 
    eval_with_chat_template, 
    generation_batch_size, 
    editing_method,
    load_edited_model,
    **kwargs
):
    """
    Evaluate and save
    """
    if editing_method in [
        "no_edit", "LoRA", "ROME", "R-ROME", "MEMIT", "AlphaEdit", "WISE", "IFMET", "CaKE", 
        "AlphaEdit_Circuit",
    ]:
        metrics = evaluate(
            model=model,
            tokenizer=tokenizer, 
            items=items, 
            alg_name=editing_method, 
            eval_with_chat_template=eval_with_chat_template,
            generation_batch_size=generation_batch_size,
            **kwargs
        )
    elif editing_method in ["MeLLo", "MeLLo_KV"]:
        # Chat models are fine-tuned on conversational data with explicit turn-taking.
        # MeLLo was designed for text-davinci-003, which is not a chat model. So, the original code does not use multi turn.
        # For fair comparison, When we evaluate MeLLo on a chat model, we evaluate by splitting the same prompt in multi turn.
        # original MeLLo is evaluated with:
        # use_kv_cache = False
        # eval_with_chat_template = "few_shot_in_user_content"
        use_kv_cache = False
        mello_eval_with_chat_template = False
        
        if editing_method == "MeLLo":
            if eval_with_chat_template:
                mello_eval_with_chat_template = "few_shot_in_user_content"
        elif editing_method == "MeLLo_KV":
            use_kv_cache = True
            if eval_with_chat_template:
                mello_eval_with_chat_template = "few_shot_in_multi_turn"
        else:
            raise NotImplementedError
        
        if editing_method == "MeLLo":
            kwargs["continuation"] = "in_user_content"
        
        metrics = evaluate_mello(
            model=model,
            tokenizer=tokenizer, 
            contriever=contriever,
            contriever_tokenizer=contriever_tokenizer,
            task_prompt=task_prompt,
            items=items,
            embeddings=embeddings,
            new_facts=new_facts,
            eval_with_chat_template=mello_eval_with_chat_template,
            generation_batch_size=generation_batch_size,
            use_kv_cache=use_kv_cache,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Evaluation method for {editing_method} not implemented.")
    
    for key in metrics.keys():
        if key not in all_metrics:
            all_metrics[key] = []
        all_metrics[key].extend(metrics[key])
        
        folder_path = f'{SAVE_METRICS_DIR}/{key}'
        os.makedirs(folder_path, exist_ok=True)
        
        metrics_str = "metrics.json" if not load_edited_model else "metrics_loaded.json"
        metrics_filepath = f'{folder_path}/{metrics_str}'
        print(f"Saving metrics to {metrics_filepath}")
        json.dump(all_metrics[key], open(metrics_filepath, 'w'), indent=4, ensure_ascii=False)
        
        res = calculate_averages(all_metrics[key])
        print(f"Results {key}:\n{json.dumps(res, indent=4)}")
        
        res_str = "res.json" if not load_edited_model else "res_loaded.json"
        res_filepath = f'{folder_path}/{res_str}'
        print(f"Saving results to {res_filepath}")
        json.dump(res, open(res_filepath, 'w'), indent=4)
    
    return metrics

def do_edit(
    model,
    tokenizer,
    data,
    editing_method,
    batch_size,
    **kwargs,
):
    if editing_method == "WISE":
        # This is from the [baseline repository](https://github.com/zjunlp/CaKE/tree/df528e13fffbd2117e6041c129285e1e146933d2)
        # The original code use ZsRE for loc_prompts and MQuAKE for edit.
        # However, when ran, the loss does not decrease.
        loc_data = json.load(open('./datasets/ZsRE/zsre_mend_train.json','r'))
        loc_data = loc_data[:7000]
        loc_index = 0
    
    indices = list(range(0, len(data), batch_size))
    if editing_method in ["LoRA", "ROME", "R-ROME", "MEMIT", "AlphaEdit", "AlphaEdit_Circuit", "WISE", "IFMET", "CaKE"]:
        all_metrics = {}
        
        for start_idx in tqdm(indices, desc="Editing the model"):
            end = min(start_idx + batch_size, len(data))
            items = data[start_idx:end]
            
            if editing_method in ["LoRA", "ROME", "R-ROME", "MEMIT", "AlphaEdit", "AlphaEdit_Circuit"]:
                model = edit(
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                    hparams=kwargs["hparams"],
                    apply_algo=kwargs["apply_algo"],
                )
            elif editing_method == "WISE":
                loc_index = edit_wise(
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                    hparams=kwargs["hparams"],
                    loc_data=loc_data,
                    loc_index=loc_index,
                    apply_algo=kwargs["apply_algo"],
                )
            elif editing_method == "IFMET":
                model = edit_ifmet(
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                    hparams_shallow=kwargs["hparams_shallow"],
                    hparams_deeper=kwargs["hparams_deeper"],
                    apply_algo=kwargs["apply_algo"],
                )
            elif editing_method == "CaKE":
                model = edit_cake(
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                )
            else:
                raise NotImplementedError(f"Editing method {editing_method} not implemented in the main loop.")
            
            if kwargs["glue_eval_steps"] > 0 and (start_idx / batch_size) % kwargs["glue_eval_steps"] == 0:
                fh = do_glue_eval(
                    save_dir=kwargs["glue_eval_save_dir"], 
                    edit_num=end, 
                    case_ids=[item['case_id'] for item in items], 
                    number_of_tests=kwargs["glue_eval_size"]
                )
                print(f"Saved GLUE evaluation results to {fh} after editing up to case_id {items[-1]['case_id']}.")
                
            if kwargs["main_eval_steps"] > 0 and (start_idx / batch_size) % kwargs["main_eval_steps"] == 0:
                metrics = do_evaluation(
                    all_metrics=all_metrics,
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                    editing_method=editing_method,
                    **kwargs
                )
                
                for key in metrics.keys():
                    print(f"Batch index={start_idx // batch_size}, key={key}:\n{json.dumps(metrics[key][0], indent=4, ensure_ascii=False)}")
    elif editing_method in ["no_edit"]:
        if kwargs["glue_eval_size"] > 0:
            fh = do_glue_eval(save_dir=kwargs["glue_eval_save_dir"], edit_num=0, case_ids=[], number_of_tests=kwargs["glue_eval_size"])
            print(f"Saved GLUE evaluation results to {fh}.")
    elif editing_method in ["MeLLo", "MeLLo_KV"]: # In-Context Learning Editing Method
        pass
    else:
        raise NotImplementedError(f"Editing method {editing_method} not implemented for editing loop.")

if __name__ == "__main__":    
    if not args.load_edited_model:
        kwargs = {}
        
        kwargs["glue_eval_save_dir"] = SAVE_METRICS_DIR
        kwargs["glue_eval_steps"] = args.glue_eval_steps
        kwargs["glue_eval_size"] = args.glue_eval_size
        
        kwargs["main_eval_steps"] = args.main_eval_steps
        kwargs["eval_with_chat_template"] = args.eval_with_chat_template
        kwargs["generation_batch_size"] = args.generation_batch_size
        kwargs["load_edited_model"] = args.load_edited_model
        if args.data_name == "MQuAKE-CF-3k-v2-clke":
            kwargs["languages"] = ["en", "de", "fr", "ja", "zh"]
            all_metrics = {language: [] for language in kwargs["languages"]}
        
        if args.editing_method in ["LoRA", "ROME", "R-ROME", "MEMIT", "AlphaEdit", "AlphaEdit_Circuit"]:
            kwargs["hparams"] = hparams
            kwargs["apply_algo"] = apply_algo
        elif args.editing_method == "WISE":
            kwargs["hparams"] = hparams
            kwargs["apply_algo"] = apply_algo
        elif args.editing_method == "IFMET":
            kwargs["hparams_shallow"] = hparams
            kwargs["hparams_deeper"] = hparams_deeper
            kwargs["apply_algo"] = apply_algo
        
        do_edit(
            model=model,
            tokenizer=tokenizer,
            data=data,
            editing_method=args.editing_method,
            batch_size=args.batch_size,
            **kwargs,
        )
        
# %%
        
if __name__ == "__main__":
    if args.save_edited_model and not args.load_edited_model:
        kwargs = {}
        if args.editing_method == "WISE":
            kwargs["hparams"] = hparams
        
        checkpoint_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            editing_method=args.editing_method,
            checkpoint_root=SAVE_CHECKPOINTS_DIR,
            edit_num=len(data),
            case_id=data[-1]["case_id"],
            **kwargs,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

# %%
    
if __name__ == "__main__":
    
    if args.main_eval_steps == 0:
        all_metrics = {}
        
        kwargs = {}
        if args.data_name == "MQuAKE-CF-3k-v2-clke":
            kwargs["languages"] = ["en", "de", "fr", "ja", "zh"]
            all_metrics = {language: [] for language in kwargs["languages"]}
        
        indices = list(range(0, len(data), args.batch_size))
        for start_idx in tqdm(indices, desc="Evaluating the edited model"):
            end = min(start_idx + args.batch_size, len(data))
            items = data[start_idx:end]
    
            metrics = do_evaluation(
                all_metrics=all_metrics,
                model=model,
                tokenizer=tokenizer,
                items=items,
                eval_with_chat_template=args.eval_with_chat_template,
                generation_batch_size=args.generation_batch_size,
                editing_method=args.editing_method,
                load_edited_model=args.load_edited_model,
                **kwargs
            )
            
            if (start_idx / args.batch_size) % 2 == 0:
                for key in metrics.keys():
                    print(f"Batch index={start_idx // args.batch_size}, key={key}:\n{json.dumps(metrics[key], indent=4, ensure_ascii=False)}")

# %%