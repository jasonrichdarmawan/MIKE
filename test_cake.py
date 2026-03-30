# %%

from EasyEdit.easyeditor import (
    MEMITHyperParams, LoRAHyperParams, WISEHyperParams, ROMEHyperParams,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from edit_utils import (
    edit, edit_ifmet, edit_mello, edit_rome, cake, edit_wise, get_sent_embeddings,
)
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import os

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
    import sys
    sys.argv = ["", "--editing_method", "MEMIT-CLKE", "--model_type", "qwen3-4b-instruct",
                "--datatype", "MQuAKE-CF-3k-v2-clke"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--model_type', default=None,type=str)
    args = parser.parse_args()

    if args.editing_method in ['MEMIT', 'MEMIT-CLKE', 'IFMET']:
        editing_hparams = MEMITHyperParams
    elif args.editing_method in ['LoRA', 'CAKE', 'Mello']:
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise ValueError(f"Unsupported editing method: {args.editing_method}")
    
    if args.editing_method in ['LoRA', 'CAKE', 'Mello']:
        hparams = editing_hparams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
    elif args.editing_method == 'IFMET':
        hparams = editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-shallow.yaml')
        hparams_s = hparams
        hparams_d = editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-deeper.yaml')
    else:
        hparams = editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml')
    
    # In-Context Learning Editing Methods (e.g., Mello)
    if args.editing_method != 'Mello':
        alg_name = hparams.alg_name
        apply_algo = ALG_DICT[alg_name]
        
# %%

if __name__ == "__main__":
    if args.editing_method == 'CAKE':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name, device_map="auto", torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
# %%

if __name__ == "__main__":
    import json
    
    data = json.load(open(f'./datasets/{args.datatype}.json','r'))
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
    
    for items in tqdm(data[:2]):
        if args.editing_method == 'CAKE':
            model, metrics = cake(model, tokenizer, items, hparams, test_generation=False)
        elif args.editing_method == 'WISE':
            metrics, loc_index = edit_wise(model, tokenizer, items, hparams, loc_data, loc_index, apply_algo, test_generation=False)
        elif args.editing_method == 'Mello':
            metrics = edit_mello(model, task_prompt, stop, tokenizer, items, hparams, contriever, contriever_tokenizer, embs, test_generation=False)
        elif args.editing_method == 'IFMET':
            model, metrics = edit_ifmet(model, tokenizer, items, hparams_s, hparams_d, apply_algo, test_generation=False)
        elif args.editing_method == 'ROME':
            model, metrics = edit_rome(model, tokenizer, items, hparams, apply_algo, test_generation=False)
        else:
            model, metrics = edit(model, tokenizer, items, hparams, alg_name, apply_algo, test_generation=False)
        print(metrics)
        all_metrics.append(metrics)
    
    res = calculate_averages(all_metrics)
    
# %%

if __name__ == "__main__":
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    json.dump(all_metrics, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_metrics.json','w'), indent=4)
    json.dump(res, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_res.json','w'), indent=4)

# %%
