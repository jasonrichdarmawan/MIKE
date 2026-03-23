from EasyEdit.easyeditor import MEMITHyperParams,LoRAHyperParams,WISEHyperParams, ROMEHyperParams
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from edit_utils import edit, edit_ifmet, edit_mello, edit_rome, cake, edit_wise, get_sent_embeddings, compute_edit_quality
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import os

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
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--model_type', default=None,type=str)
    parser.add_argument('--lang', default=None, type=str)
    parser.add_argument('--multilingual', action='store_true', help='Edit in English, evaluate in all 5 languages')
    args = parser.parse_args()

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
    elif args.editing_method == 'PreEdit':
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
    else:
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml')
    
    if args.editing_method not in ['Mello', 'PreEdit']:
        alg_name = hparams.alg_name
        apply_algo = ALG_DICT[alg_name]

    MODEL_PATH = hparams.model_name
    if args.editing_method == 'CAKE':
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    # 添加FFN层的LoRA配置
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    import json
    data = json.load(open(f'./datasets/{args.datatype}.json','r', encoding='utf-8'))
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

    hparams.lang = args.lang

    # Checkpoint: load existing results for resume
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    lang_tag = f"_{args.lang}" if args.lang else ""
    metrics_path = f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}{lang_tag}_metrics.json'
    if os.path.exists(metrics_path):
        all_metrics = json.load(open(metrics_path, 'r', encoding='utf-8'))
        start_idx = len(all_metrics)
        print(f"Resuming from item {start_idx} ({start_idx} already completed)")
    else:
        all_metrics = []
        start_idx = 0

    if args.editing_method == 'PreEdit':
        from edit_utils import check_answer_batch
        EVAL_BATCH_SIZE = 20  # Process 16 cases at once for faster evaluation

        suffix = f"_{args.lang}" if args.lang and args.lang != "en" else ""
        single_hops_key = f"new_single_hops{suffix}"
        new_answer_key = f"new_answer{suffix}"
        new_answer_alias_key = f"new_answer_alias{suffix}"
        cloze_question_key = f"cloze_question{suffix}"
        questions_key = f"questions{suffix}"
        device = f"cuda:{hparams.device}"

        remaining_data = data[start_idx:]
        for batch_start in tqdm(range(0, len(remaining_data), EVAL_BATCH_SIZE), 
                                initial=start_idx // EVAL_BATCH_SIZE, 
                                total=(len(data) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE,
                                desc="PreEdit batched eval"):
            batch_items = remaining_data[batch_start:batch_start + EVAL_BATCH_SIZE]

            # Collect ALL questions from all items in this batch
            all_questions = []
            all_answers = []
            question_map = []  # (item_idx, category, sub_idx)

            for item_idx, item in enumerate(batch_items):
                # hop_wise questions
                for hop_idx, hop in enumerate(item.get(single_hops_key, [])):
                    ans = list(hop.get('answer_alias', []))
                    ans.append(hop.get('answer', ''))
                    # cloze
                    all_questions.append(hop.get('cloze', ''))
                    all_answers.append(ans)
                    question_map.append((item_idx, 'hop', hop_idx, 'cloze'))
                    # question
                    all_questions.append(hop.get('question', ''))
                    all_answers.append(ans)
                    question_map.append((item_idx, 'hop', hop_idx, 'question'))

                # accuracy questions
                answer = list(item.get(new_answer_alias_key, []))
                answer.append(item.get(new_answer_key, ''))
                # cloze
                all_questions.append(item.get(cloze_question_key, ''))
                all_answers.append(answer)
                question_map.append((item_idx, 'acc', 0, 'cloze'))
                # full questions
                for q_idx, q in enumerate(item.get(questions_key, [])):
                    all_questions.append(q)
                    all_answers.append(answer)
                    question_map.append((item_idx, 'acc', q_idx + 1, 'question'))

            # Batch inference all questions at once
            if all_questions:
                batch_results = check_answer_batch(model, all_questions, tokenizer, all_answers, device, max_new_tokens=100)
            else:
                batch_results = []

            # Distribute results back to each item
            for item_idx, item in enumerate(batch_items):
                metrics = {
                    'case_id': item['case_id'],
                    'requested_rewrite': item['requested_rewrite'],
                    'time': 0,
                    'post': {
                        'hop_wise': [],
                        'accuracy': [],
                        'hop_wise_predictions': [],
                        'accuracy_predictions': []
                    }
                }

                # Collect hop_wise results
                hop_data = {}
                for map_entry, (matched, gen_text) in zip(question_map, batch_results):
                    if map_entry[0] != item_idx:
                        continue
                    if map_entry[1] == 'hop':
                        hop_idx = map_entry[2]
                        if hop_idx not in hop_data:
                            hop_data[hop_idx] = {'metrics': [], 'predictions': []}
                        hop_data[hop_idx]['metrics'].append(matched)
                        hop_data[hop_idx]['predictions'].append({
                            'question': all_questions[question_map.index(map_entry)],
                            'generated_text': gen_text,
                            'answer': all_answers[question_map.index(map_entry)][-1]
                        })
                    elif map_entry[1] == 'acc':
                        metrics['post']['accuracy'].append(matched)
                        metrics['post']['accuracy_predictions'].append({
                            'question': all_questions[question_map.index(map_entry)],
                            'generated_text': gen_text,
                            'answer': item.get(new_answer_key, '')
                        })

                for hop_idx in sorted(hop_data.keys()):
                    metrics['post']['hop_wise'].append(hop_data[hop_idx]['metrics'])
                    metrics['post']['hop_wise_predictions'].append(hop_data[hop_idx]['predictions'])

                all_metrics.append(metrics)

            # Save checkpoint after each batch
            json.dump(all_metrics, open(metrics_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    else:
        for item in tqdm(data[start_idx:], initial=start_idx, total=len(data)):
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
            else:
                model, metrics = edit(model, tokenizer, item, hparams, alg_name, apply_algo, test_generation=False)
            print(metrics)
            all_metrics.append(metrics)
            json.dump(all_metrics, open(metrics_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    res = calculate_averages(all_metrics)
    json.dump(res, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}{lang_tag}_res.json','w', encoding='utf-8'), indent=4, ensure_ascii=False)

    # ============================================================
    # Multilingual mode: edit in English, evaluate in all 5 languages
    # Usage: add --multilingual flag
    # Example: python test_cake.py --editing_method MEMIT --datatype MQuAKE-CF-3k-v2-clke --model_type qwen3-4b --multilingual
    # Original single-language code above is preserved unchanged.
    # ============================================================
    if args.multilingual and args.editing_method not in ['PreEdit']:
        from edit_utils import (edit_multilingual, edit_rome_multilingual, 
                                cake_multilingual, edit_wise_multilingual)

        EVAL_LANGS = ['en', 'de', 'fr', 'ja', 'zh']
        print(f"\n{'='*60}")
        print(f"Multilingual mode: edit in English, evaluate in {EVAL_LANGS}")
        print(f"{'='*60}\n")

        # Initialize per-language metrics and checkpoint paths
        lang_all_metrics = {}
        lang_metrics_paths = {}
        for lang in EVAL_LANGS:
            path = f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{lang}_metrics.json'
            lang_metrics_paths[lang] = path
            if os.path.exists(path):
                lang_all_metrics[lang] = json.load(open(path, 'r', encoding='utf-8'))
            else:
                lang_all_metrics[lang] = []

        # Resume from shortest completed language
        completed_counts = [len(lang_all_metrics[lang]) for lang in EVAL_LANGS]
        ml_start_idx = min(completed_counts)
        if ml_start_idx > 0:
            for lang in EVAL_LANGS:
                lang_all_metrics[lang] = lang_all_metrics[lang][:ml_start_idx]
            print(f"Resuming multilingual eval from item {ml_start_idx}")

        # Reload model fresh for multilingual eval
        print("Reloading model for multilingual evaluation...")
        del model
        torch.cuda.empty_cache()
        if args.editing_method == 'CAKE':
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Re-initialize WISE loc_data if needed
        if args.editing_method == 'WISE':
            loc_index = 0

        data = data[:100]
        for item in tqdm(data[ml_start_idx:], initial=ml_start_idx, total=len(data), desc="Multilingual edit+eval"):
            # Edit and evaluate in all 5 languages
            if args.editing_method == 'CAKE':
                model, lang_results = cake_multilingual(model, tokenizer, item, hparams, test_generation=False)
            elif args.editing_method == 'WISE':
                lang_results, loc_index = edit_wise_multilingual(model, tokenizer, item, hparams, loc_data, loc_index, apply_algo, test_generation=False)
            elif args.editing_method == 'ROME':
                model, lang_results = edit_rome_multilingual(model, tokenizer, item, hparams, apply_algo, test_generation=False)
            else:
                model, lang_results = edit_multilingual(model, tokenizer, item, hparams, alg_name, apply_algo, test_generation=False)

            # Save results for each language
            for lang in EVAL_LANGS:
                lang_all_metrics[lang].append(lang_results[lang])
                json.dump(lang_all_metrics[lang], open(lang_metrics_paths[lang], 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

            print(f"Item {item['case_id']} done")

        # Save final res for each language
        for lang in EVAL_LANGS:
            lang_res = calculate_averages(lang_all_metrics[lang])
            lang_res_path = f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{lang}_res.json'
            json.dump(lang_res, open(lang_res_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
            print(f"Language {lang}: {lang_res}")
