try:
    from EasyEdit.easyeditor import MEMITHyperParams,LoRAHyperParams,WISEHyperParams, ROMEHyperParams
    from EasyEdit.easyeditor.util.alg_dict import *
except Exception:
    pass
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from edit_utils import edit, edit_ifmet, edit_mello, edit_rome, cake, edit_wise, get_sent_embeddings, compute_edit_quality
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# GLUE is run at these edit counts for AlphaEdit (1500 is intentionally skipped)
ALPHAEDIT_GLUE_CHECKPOINTS = {0, 500, 1000, 2000, 2500, 3000}

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None, type=str,
                        help='Dataset name. For AlphaEdit defaults to counterfact. '
                             'For all other methods required (loads ./datasets/<datatype>-cake.json).')
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument('--eval_interval', default=500, type=int,
                        help='Run GLUE eval every N edits for non-AlphaEdit methods '
                             '(0 = only after all edits). AlphaEdit always uses fixed checkpoints.')
    parser.add_argument('--alphaedit_dir', default='./alphaedit_repo', type=str,
                        help='Path to AlphaEdit repo (required for AlphaEdit method and GLUEEval)')
    parser.add_argument('--model_path', default=None, type=str,
                        help='HuggingFace model path (required for AlphaEdit method)')
    args = parser.parse_args()

    cake_dir = os.path.dirname(os.path.abspath(__file__))
    alphaedit_dir = os.path.abspath(args.alphaedit_dir)

    # AlphaEdit repo must be on sys.path for AlphaEdit module imports
    # and for util.perplexity used by GLUEEval.
    # cake_dir inserted last so it stays at position 0 (patched glue_eval takes priority).
    sys.path.insert(0, alphaedit_dir)
    sys.path.insert(0, cake_dir)

    if args.editing_method == 'AlphaEdit':
        # globals.yml is opened at import time using a relative path
        os.chdir(alphaedit_dir)
        from AlphaEdit import apply_AlphaEdit_to_model, AlphaEditHyperParams
        from AlphaEdit.AlphaEdit_main import get_cov
        from glue_eval.glue_eval import GLUEEval
        os.chdir(cake_dir)
    elif args.eval_interval > 0:
        from glue_eval.glue_eval import GLUEEval

    if args.editing_method == 'AlphaEdit':
        hparams = AlphaEditHyperParams.from_json(
            os.path.join(alphaedit_dir, 'hparams', 'AlphaEdit', f'{args.model_type}.json')
        )
        MODEL_PATH = args.model_path
    else:
        if args.editing_method == 'MEMIT' or args.editing_method == 'IFMET':
            editing_hparams = MEMITHyperParams
        elif args.editing_method == 'LoRA':
            editing_hparams = LoRAHyperParams
        elif args.editing_method == 'WISE':
            editing_hparams = WISEHyperParams
        elif args.editing_method == 'ROME':
            editing_hparams = ROMEHyperParams
        else:
            editing_hparams = LoRAHyperParams
        if args.editing_method == 'CAKE' or args.editing_method == 'Mello':
            hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
        elif args.editing_method == 'IFMET':
            hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-shallow.yaml')
            hparams_s=hparams
            hparams_d=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-deeper.yaml')
        else:
            hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml')

        if args.editing_method != 'Mello':
            alg_name = hparams.alg_name
            apply_algo = ALG_DICT[alg_name]

        MODEL_PATH = hparams.model_name

    if args.editing_method == 'AlphaEdit':
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    elif args.editing_method == 'CAKE':
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    import json

    # ---------- data loading ----------
    if args.editing_method == 'AlphaEdit':
        # Default dataset is CounterFact; --datatype overrides.
        alphaedit_dtype = (args.datatype or 'counterfact').lower()
        if alphaedit_dtype == 'counterfact':
            os.chdir(alphaedit_dir)
            from dsets import CounterFactDataset
            _ds = CounterFactDataset("./data")
            data = [_ds[i] for i in range(min(3000, len(_ds)))]
            os.chdir(cake_dir)
        elif alphaedit_dtype == 'zsre':
            os.chdir(alphaedit_dir)
            from dsets import MENDQADataset
            _ds = MENDQADataset("./data", tok=tokenizer)
            data = [_ds[i] for i in range(min(3000, len(_ds)))]
            os.chdir(cake_dir)
        else:
            # CaKE-format dataset (e.g. MQuAKE)
            data = json.load(open(f'./datasets/{args.datatype}-cake.json', 'r'))[:3000]
        # Whether dataset is in CaKE multi-hop format (needs compute_edit_quality)
        alphaedit_cake_format = alphaedit_dtype not in ('counterfact', 'zsre')
    else:
        data = json.load(open(f'./datasets/{args.datatype}-cake.json','r'))

    # ---------- method-specific setup ----------
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
    elif args.editing_method == 'AlphaEdit':
        # key_dim: shape[1] for LLaMA/GPT-J/Phi (nn.Linear), shape[0] for GPT-2 (Conv1D)
        layer_name = hparams.rewrite_module_tmp.format(hparams.layers[0]) + '.weight'
        W_out = dict(model.named_parameters())[layer_name]
        key_dim = W_out.shape[1]

        # Null-space projection matrices P (one per edited layer, expensive to compute)
        P = torch.zeros(len(hparams.layers), key_dim, key_dim)
        print("Computing null-space projection matrices...")
        os.chdir(alphaedit_dir)
        for i, layer in enumerate(hparams.layers):
            cov = get_cov(
                model,
                tokenizer,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
                force_recompute=False,
            ).cpu()
            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            small_singular_indices = (S < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
            P[i] = U[:, small_singular_indices] @ U[:, small_singular_indices].T
        os.chdir(cake_dir)

        # Covariance cache accumulated across all edits
        cache_c = torch.zeros(len(hparams.layers), key_dim, key_dim, device="cpu")

    all_metrics = []
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    edit_count = 0

    # ---------- AlphaEdit baseline GLUE (before any edits) ----------
    if args.editing_method == 'AlphaEdit' and 0 in ALPHAEDIT_GLUE_CHECKPOINTS:
        glue_record = (f'{args.metrics_save_dir}/AlphaEdit_{args.model_type}'
                       f'_{alphaedit_dtype}_glue_0.json')
        glue_eval = GLUEEval(model, tokenizer, number_of_tests=100)
        glue_results = {'edit_num': 0}
        glue_results = glue_eval.evaluate(
            glue_results,
            record_path=glue_record,
            sst_flag=True, cola_flag=True, mrpc_flag=True,
            rte_flag=True, nli_flag=True, mmlu_flag=True,
        )
        json.dump(glue_results, open(glue_record, 'w'), indent=4)
        print("GLUE @ 0 edits (baseline):", glue_results)

    # ---------- main editing loop ----------
    for item in tqdm(data):
        if args.editing_method == 'CAKE':
            model, metrics = cake(model, tokenizer, item, hparams, test_generation=False)
        elif args.editing_method == 'WISE':
            metrics, loc_index = edit_wise(model, tokenizer, item, hparams, loc_data, loc_index, apply_algo, test_generation=False)
        elif args.editing_method == 'Mello':
            metrics = edit_mello(model, task_prompt, stop, tokenizer, item, hparams, contriever, contriever_tokenizer, embs, test_generation=False)
        elif args.editing_method == 'IFMET':
            model, metrics = edit_ifmet(model, tokenizer, item, hparams_s, hparams_d, apply_algo, test_generation=False)
        elif args.editing_method == 'ROME':
            model, metrics = edit_rome(model, tokenizer, item, hparams, apply_algo, test_generation=False)
        elif args.editing_method == 'AlphaEdit':
            rw = item['requested_rewrite']
            if isinstance(rw, list):
                # CaKE MQuAKE format: list of rewrites, target_new is a dict {'str': ...}
                requests = rw
                for r in requests:
                    if isinstance(r.get('target_new'), dict):
                        r['target_new'] = r['target_new']['str']
            else:
                # CounterFact / ZsRE format: single rewrite dict, passed as-is
                requests = [rw]
            os.chdir(alphaedit_dir)
            model, cache_c = apply_AlphaEdit_to_model(
                model, tokenizer, requests, hparams, cache_c=cache_c, P=P
            )
            os.chdir(cake_dir)
            if alphaedit_cake_format:
                metrics = {
                    'case_id': item['case_id'],
                    'requested_rewrite': requests,
                    'post': compute_edit_quality(model, tokenizer, item, hparams),
                }
            else:
                metrics = {'case_id': item.get('case_id', edit_count)}
        else:
            model, metrics = edit(model, tokenizer, item, hparams, alg_name, apply_algo, test_generation=False)
        print(metrics)
        all_metrics.append(metrics)
        edit_count += 1

        # ---------- GLUE evaluation trigger ----------
        if args.editing_method == 'AlphaEdit':
            run_glue = edit_count in ALPHAEDIT_GLUE_CHECKPOINTS
            dtype_label = alphaedit_dtype
        else:
            run_glue = args.eval_interval > 0 and edit_count % args.eval_interval == 0
            dtype_label = args.datatype

        if run_glue:
            glue_record = (f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}'
                           f'_{dtype_label}_glue_{edit_count}.json')
            glue_eval = GLUEEval(model, tokenizer, number_of_tests=100)
            glue_results = {'edit_num': edit_count}
            glue_results = glue_eval.evaluate(
                glue_results,
                record_path=glue_record,
                sst_flag=True, cola_flag=True, mrpc_flag=True,
                rte_flag=True, nli_flag=True, mmlu_flag=True,
            )
            json.dump(glue_results, open(glue_record, 'w'), indent=4)
            print(f"GLUE @ {edit_count} edits:", glue_results)

    # ---------- save results ----------
    save_label = alphaedit_dtype if args.editing_method == 'AlphaEdit' else args.datatype
    json.dump(all_metrics, open(
        f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{save_label}_metrics.json', 'w'
    ), indent=4)

    # calculate_averages needs CaKE multi-hop format; skip for AlphaEdit counterfact/zsre
    if args.editing_method != 'AlphaEdit' or alphaedit_cake_format:
        res = calculate_averages(all_metrics)
        json.dump(res, open(
            f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{save_label}_res.json', 'w'
        ), indent=4)
