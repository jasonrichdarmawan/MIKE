# %%

from EasyEdit.easyeditor import (
    MEMITHyperParams, LoRAHyperParams, WISEHyperParams, ROMEHyperParams,
    AlphaEditHyperParams, AlphaEditCircuitHyperParams, R_ROMEHyperParams,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm import tqdm
from edit_utils import (
    edit, edit_ifmet, edit_mello, edit_rome, cake, edit_wise, get_sent_embeddings,
)
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import os
import logging
from utils import StreamHandler, FileHandler, BufferingLoggerWriter
import sys

# %%

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
            case_hop_true_count += sum(1 for x in hop_pair if x)
            total_hops += len(hop_pair)
        
        case_hop_average = case_hop_true_count / total_hops
        hop_wise_case_averages.append(case_hop_average)
        
        accuracy_true_count = sum(1 for x in accuracy if x)
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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sys.argv = ["", 
        "--editing_method", "AlphaEdit_Circuit", 
        "--sequential_editing",
        "--model_type", "qwen3-4b-instruct",
        "--dtype", "bfloat16",
        "--apply_chat_template",
        "--batch_size", "2",
        "--sample_size", "4",
        "--data_name", "MQuAKE-CF-3k-v2-clke",
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--sequential_editing', action='store_true')
    parser.add_argument('--data_name', default=None, type=str)
    parser.add_argument('--model_type', default=None,type=str)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--apply_chat_template', action='store_true')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--sample_size", default=3000, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    args = parser.parse_args()
    
    SAVE_DIR = f'{args.metrics_save_dir}/{args.model_type}/{args.data_name}/s{args.sample_size}_b{args.batch_size}/'

    if args.editing_method == 'no_edit':
        hparams = AlphaEditCircuitHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit_Circuit/{args.model_type}.yaml')
    elif args.editing_method in ['MEMIT']:
        hparams = MEMITHyperParams.from_hparams(f'./EasyEdit/hparams/MEMIT/{args.model_type}.yaml')
    elif args.editing_method in ['IFMET']:
        hparams_s = MEMITHyperParams.from_hparams(f'./EasyEdit/hparams/IFMET/{args.model_type}-shallow.yaml')
        hparams_d = MEMITHyperParams.from_hparams(f'./EasyEdit/hparams/IFMET/{args.model_type}-deeper.yaml')
    elif args.editing_method in ['LoRA', 'CAKE', 'Mello']:
        hparams = LoRAHyperParams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
    elif args.editing_method == 'WISE':
        hparams = WISEHyperParams.from_hparams(f'./EasyEdit/hparams/WISE/{args.model_type}.yaml')
    elif args.editing_method == 'ROME':
        hparams = ROMEHyperParams.from_hparams(f'./EasyEdit/hparams/ROME/{args.model_type}.yaml')
    elif args.editing_method == "R-ROME":
        hparams = R_ROMEHyperParams.from_hparams(f'./EasyEdit/hparams/R-ROME/{args.model_type}.yaml')
    elif args.editing_method == "AlphaEdit":
        hparams = AlphaEditHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit/{args.model_type}.yaml')
    elif args.editing_method == "AlphaEdit_Circuit":
        hparams = AlphaEditCircuitHyperParams.from_hparams(f'./EasyEdit/hparams/AlphaEdit_Circuit/{args.model_type}.yaml')
    else:
        raise ValueError(f"Unsupported editing method: {args.editing_method}")
    
    # In-Context Learning Editing Methods (e.g., Mello)
    if args.editing_method not in ["no_edit", 'Mello']:
        alg_name = hparams.alg_name
        apply_algo = ALG_DICT[alg_name]
        
# %%

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)-9s %(message)s')
    
    sh = StreamHandler()
    sh.setFormatter(fmt)
    
    fh = FileHandler(f"{SAVE_DIR}/output.log", mode='w')
    fh.setFormatter(fmt)
    
    logger = logging.getLogger("CaKE_Test")
    logger.propagate = False  # Prevent log messages from being propagated to the root logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    # Redirect stdout to the logger
    sys.stdout = BufferingLoggerWriter(logger, logging.INFO)
    sys.stderr = BufferingLoggerWriter(logger, logging.WARNING)
    
# %%

if __name__ == "__main__":
    dtype = getattr(torch, args.dtype)
    max_memory = {i: "8GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name, device_map="auto", max_memory=max_memory, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
# %%

if __name__ == "__main__":
    data = json.load(open(f'./datasets/{args.data_name}.json','r'))
    data = data[:args.sample_size]
    if args.editing_method == 'WISE':
        loc_data = json.load(open('./datasets/ZsRE/zsre_mend_train.json','r'))
        loc_data = loc_data[:7000]
        loc_index = 0
    elif args.editing_method == 'Mello':
        contriever = AutoModel.from_pretrained("facebook/contriever-msmarco", device_map=f"cuda:{hparams.device}")
        contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        with open('./prompts/MeLLo-prompt.txt', 'r') as f:
            task_prompt = f.read()
        stop = ["Retrieved fact:"]
        new_facts = set()
        for d in data:
            for r in d["requested_rewrite"]:
                new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
        new_facts = list(new_facts)
        embs = get_sent_embeddings(new_facts, contriever, contriever_tokenizer, hparams.device)

# %%

if __name__ == "__main__":
    all_metrics = []
    
    indices = list(range(0, len(data), args.batch_size))
    for start_idx in tqdm(indices, desc="Processing batches"):
        items = data[start_idx:start_idx + args.batch_size]
        if args.editing_method == "no_edit":
            model, metrics = edit(
                model, tokenizer, items, hparams, alg_name=None, apply_algo=None, 
                apply_chat_template=args.apply_chat_template, restore_weights=False, 
                test_generation=False
            )
        elif args.editing_method == 'CAKE':
            model, metrics = cake(
                model, tokenizer, items, hparams, apply_chat_template=args.apply_chat_template, 
                restore_weights=args.sequential_editing, test_generation=False
            )
        elif args.editing_method == 'WISE':
            metrics, loc_index = edit_wise(
                model, tokenizer, items, hparams, loc_data, loc_index, apply_algo, 
                apply_chat_template=args.apply_chat_template, restore_weights=args.sequential_editing, 
                test_generation=False
            )
        elif args.editing_method == 'Mello':
            metrics = edit_mello(
                model, task_prompt, stop, tokenizer, items, hparams, contriever, contriever_tokenizer, 
                embs, apply_chat_template=args.apply_chat_template, test_generation=False
            )
        elif args.editing_method == 'IFMET':
            model, metrics = edit_ifmet(
                model, tokenizer, items, hparams_s, hparams_d, apply_algo, 
                apply_chat_template=args.apply_chat_template, restore_weights=args.sequential_editing, 
                test_generation=False
            )
        elif args.editing_method == 'ROME':
            model, metrics = edit_rome(
                model, tokenizer, items, hparams, apply_algo, 
                apply_chat_template=args.apply_chat_template, restore_weights=args.sequential_editing, 
                test_generation=False
            )
        else:
            model, metrics = edit(
                model, tokenizer, items, hparams, alg_name, apply_algo, 
                apply_chat_template=args.apply_chat_template, restore_weights=args.sequential_editing, 
                test_generation=False
            )
        
        if start_idx % args.batch_size == 0:
            print(f"Batch index: {start_idx // args.batch_size}:\n{json.dumps(metrics[0], indent=4)}")
            
        all_metrics.extend(metrics)
    
    res = calculate_averages(all_metrics)
    print(f"Results:\n{json.dumps(res, indent=4)}")
    
# %%

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    metrics_filepath = f'{SAVE_DIR}/{args.editing_method}_metrics.json'
    res_filepath = f'{SAVE_DIR}/{args.editing_method}_res.json'
    print(f"Saving metrics to {metrics_filepath}")
    print(f"Saving results to {res_filepath}")
    json.dump(all_metrics, open(metrics_filepath, 'w'), indent=4)
    json.dump(res, open(res_filepath, 'w'), indent=4)

# %%
