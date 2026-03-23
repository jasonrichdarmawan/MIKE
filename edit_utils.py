import torch
from typing import List, Dict, Union
from tqdm import tqdm
from time import time
from peft import LoraConfig, TaskType, get_peft_model
from EasyEdit.easyeditor.util import nethook
import random
from datasets import Dataset
from transformers import TrainingArguments, Trainer, StoppingCriteria, StoppingCriteriaList


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def _extract_answer_text(prediction):
    """Extract answer from <answer> tags, or return raw prediction if no tags found."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", prediction, re.IGNORECASE | re.DOTALL)
    text = m.group(1) if m else prediction
    return text.strip()

def check_answer_in_pred(pred, answers):
    pred = _extract_answer_text(pred).lower()
    for a in answers:
        a = a.strip().lower()
        if not a:
            continue
        if a in pred or pred.startswith(a) or pred == a:
            return True
    return False


def _supports_chat_template(tokenizer):
    return hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None


def _format_chat_prompt(prompt, tokenizer):
    if not _supports_chat_template(tokenizer):
        return prompt
    system_content = (
        "1. Only respond with the answer to the question. Do not include any explanations or additional text."
        "\n2. Wrap your final answer in XML tags like this: <answer>ANSWER_HERE</answer>."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _format_chat_train_pair(text, target, tokenizer):
    """
    Example:
    - text: ""If Maria lives in a mansion, Fernando Santos resides in a townhouse, and Jose lives in a bungalow, then the person who resides in a townhouse is a citizen of which country?"
    - target: "The person who resides in a townhouse is a citizen of the United Kingdom."
    """
    if not _supports_chat_template(tokenizer):
        full_text = f"{text} {target}"
        prompt_text = text
        return full_text, prompt_text

    full_messages = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": target},
    ]
    prompt_messages = [{"role": "user", "content": text}]

    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return full_text, prompt_text

def get_sent_embeddings(sents, contriever, tok, device, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, device, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices

def create_lora_model(
    model,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
):
    """
    Creates a LoRA-wrapped causal language model over all layers.
    """
    # Create LoRA config
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,  # for generative text models
    )
    
    # 打印将应用LoRA的模块信息
    print(f"Applying LoRA to the following modules: {target_modules}")
    # Wrap the model with LoRA adapters
    lora_model = get_peft_model(model, peft_config)
    print("LoRA model created.")
    return lora_model



class MultiStopCriteria(StoppingCriteria):
    def __init__(self, stop_token_sequences):
        self.stop_sequences = stop_token_sequences

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最新生成的 token 是否匹配任意停止序列
        for stop_seq in self.stop_sequences:
            if input_ids.shape[1] >= len(stop_seq):
                if torch.all(input_ids[0, -len(stop_seq):] == torch.tensor(stop_seq, device=input_ids.device)):
                    return True  # 触发停止
        return False

def call_model(prompt, stop, model, tokenizer):
    # ==== 关键模型适配 ====
    # 针对不同模型设置特殊 token（以 Llama3 和 Qwen2.5 为例）
    if "llama3" in model.config.model_type.lower():
        # Llama3-instruct 需要添加系统提示模板
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.eos_token = "<|eot_id|>"
    elif "qwen" in model.config.model_type.lower():
        # Qwen2.5-instruct 使用 <|im_start|> 模板
        full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokenizer.pad_token = tokenizer.eos_token  # 确保设置 pad_token
    else:
        full_prompt = prompt

    # ==== 停止词处理 ====
    # 将停止词编码为 token ID 序列（处理多 token 情况）
    stop_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop]
    stop_criteria = MultiStopCriteria(stop_sequences)

    # ==== 模型输入编码 ====
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(model.device)

    # ==== 生成参数配置 ====
    generate_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "stopping_criteria": StoppingCriteriaList([stop_criteria]),
        "pad_token_id": tokenizer.eos_token_id  # 重要：避免生成中断错误
    }

    # ==== 执行生成 ====
    with torch.no_grad():
        outputs = model.generate(**generate_args)

    # ==== 解码与后处理 ====
    # 截取新生成部分（排除原始 prompt）
    new_tokens = outputs[0, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 二次清理停止词（防止分词差异导致残留）
    for stop_word in stop:
        if stop_word in generated_text:
            generated_text = generated_text.split(stop_word)[0].strip()
            break  # 遇到第一个停止词即终止
    
    return generated_text

def mello(task_prompt, q, model, tokenizer, stop, contriever, contriever_tokenizer, embs, answer, device):
    found_ans = False
    prompt = task_prompt + "\n\nQustion: " + q
    print('*********************************')
    for i in range(4):
    # prompt the model to generate a subquestion and a tentative answer
        gen = call_model(prompt, stop, model, tokenizer)
        last_sent = gen.strip().split('\n')[-1]
        # if final answer is there, get the answer and exit
        if last_sent.startswith('Final answer: '):
            found_ans = True
            ans = last_sent[len("Final answer: "):]
            break
        # otherwise, extract the generated subquestion
        if len(gen.strip().split('\n')) < 2:
            break # failed case
        subquestion = gen.strip().split('\n')[-2]
        if not subquestion.startswith('Subquestion: '):
            break # failed case
        subquestion = subquestion[len("Subquestion: "):]

        # retrieve an edited fact using the generated subquestion
        fact_ids = retrieve_facts(subquestion, embs, contriever, contriever_tokenizer, device)
        fact_sent = new_facts[fact_ids[0]]

        # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
        prompt = prompt + '\n' + gen + '\nRetrieved fact: ' + fact_sent + '.'
        print(f'{i}:{gen}')
        print(f'{i}:{fact_sent}')
    
    prompt = prompt + gen
    if not found_ans:
        return False, prompt
    return check_answer_in_pred(ans, answer), prompt


def preprocess_function(examples, tokenizer):
    """
    Caveat:
    - CaKE's implementation do not turn padded positions into -100
    - CaKE's implementation do not use chat-template formatting, and do not mask the prompt part with -100
    
    Combine the `text` and `target` fields in each `item_case_examples`.
    
    examples: dicts-of-lists
    """
    all_texts = []
    all_targets = []
    
    for item in examples['item_case_examples']:
        """
        example['text']: "Fernando Santos is a citizen of"
        example['target']: "United Kingdom"
        """
        for example in item:
            all_texts.append(example['text'])
            all_targets.append(example['target'])
            
    # Build training pairs first so chat-template formatting can keep alignment.
    pairs = list(zip(all_texts, all_targets))
    random.shuffle(pairs)
    learning_texts = []
    learning_targets = []
    for item in examples['learning_examples']:
        """
        example['text']: "If Maria lives in a mansion, Fernando Santos resides in a townhouse, and Jose lives in a bungalow, then the person who resides in a townhouse is a citizen of which country?"
        example['target']: "The person who resides in a townhouse is a citizen of the United Kingdom."
        """
        for example in item:
            learning_texts.append(example['text'])
            learning_targets.append(example['target'])
    learning_pairs = list(zip(learning_texts, learning_targets))
    random.shuffle(learning_pairs)

    # Tokenize with/without chat template using consistent prompt/target boundaries.
    all_pairs = pairs + learning_pairs
    full_sequences = []
    prompt_sequences = []
    for text, target in all_pairs:
        full_text, prompt_text = _format_chat_train_pair(text, target, tokenizer)
        full_sequences.append(full_text)
        prompt_sequences.append(prompt_text)

    model_inputs = tokenizer(
        full_sequences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np",
        add_special_tokens=not _supports_chat_template(tokenizer),
    )
    prompt_inputs = tokenizer(
        prompt_sequences,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors=None,
        add_special_tokens=not _supports_chat_template(tokenizer),
    )

    labels = model_inputs["input_ids"].copy()
    for idx, prompt_ids in enumerate(prompt_inputs["input_ids"]):
        prefix_length = min(len(prompt_ids), labels.shape[1])
        labels[idx, :prefix_length] = -100
    labels[model_inputs["attention_mask"] == 0] = -100
    model_inputs['labels'] = labels
    return model_inputs

def check_answer(model, question, tokenizer, answer, device, max_new_tokens=100):
    """Check if the model can answer the question correctly. Uses structured prompt."""
    prompt = _format_chat_prompt(question, tokenizer)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=not _supports_chat_template(tokenizer),
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return check_answer_in_pred(generated_text, answer), generated_text

def check_answer_batch(model, questions, tokenizer, answers_list, device, max_new_tokens=100):
    """
    Batched version of check_answer. Process multiple questions at once for speed.
    questions: list of prompt strings
    answers_list: list of lists of acceptable answers
    Returns: list of (bool, str) tuples
    """
    prompts = [_format_chat_prompt(q, tokenizer) for q in questions]
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=not _supports_chat_template(tokenizer),
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    results = []
    for idx in range(len(questions)):
        generated_text = tokenizer.decode(outputs[idx][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        matched = check_answer_in_pred(generated_text, answers_list[idx])
        results.append((matched, generated_text))
    return results

def compute_edit_quality(model, tokenizer, edit_item, hparams, test_generation=False, lang=None):
    if lang is None:
        lang = getattr(hparams, 'lang', None)
    # Multi-lingual field selection
    suffix = f"_{lang}" if lang and lang != "en" else ""
    single_hops_key = f"new_single_hops{suffix}"
    new_answer_key = f"new_answer{suffix}"
    new_answer_alias_key = f"new_answer_alias{suffix}"
    cloze_question_key = f"cloze_question{suffix}"
    questions_key = f"questions{suffix}"

    metrics = {
        'hop_wise': [],
        'accuracy': [],
        'hop_wise_predictions': [],
        'accuracy_predictions': []
    }
    device = f"cuda:{hparams.device}"

    # ---- Batched hop_wise evaluation ----
    hop_questions = []
    hop_answers = []
    hop_indices = []  # track which hop each question belongs to
    for hop_idx, i in enumerate(edit_item.get(single_hops_key, [])):
        ans = list(i.get('answer_alias', []))
        ans.append(i.get('answer', ''))
        # cloze question
        hop_questions.append(i.get('cloze', ''))
        hop_answers.append(ans)
        hop_indices.append((hop_idx, 'cloze'))
        # full question
        hop_questions.append(i.get('question', ''))
        hop_answers.append(ans)
        hop_indices.append((hop_idx, 'question'))

    if hop_questions:
        hop_results = check_answer_batch(model, hop_questions, tokenizer, hop_answers, device, max_new_tokens=100)
        # Organize results back into hop_wise structure
        hop_data = {}
        for (hop_idx, q_type), (matched, gen_text), question, answers in zip(hop_indices, hop_results, hop_questions, hop_answers):
            if hop_idx not in hop_data:
                hop_data[hop_idx] = {'metrics': [], 'predictions': []}
            hop_data[hop_idx]['metrics'].append(matched)
            hop_data[hop_idx]['predictions'].append({
                "question": question,
                "generated_text": gen_text,
                "answer": answers[-1]  # last one is the main answer
            })
        for hop_idx in sorted(hop_data.keys()):
            metrics['hop_wise'].append(hop_data[hop_idx]['metrics'])
            metrics['hop_wise_predictions'].append(hop_data[hop_idx]['predictions'])

    # ---- Batched accuracy evaluation ----
    answer = list(edit_item.get(new_answer_alias_key, []))
    answer.append(edit_item.get(new_answer_key, ''))

    acc_questions = []
    acc_answers = []
    # cloze question
    acc_questions.append(edit_item.get(cloze_question_key, ''))
    acc_answers.append(answer)
    # full questions
    for i in edit_item.get(questions_key, []):
        acc_questions.append(i)
        acc_answers.append(answer)

    if acc_questions:
        acc_results = check_answer_batch(model, acc_questions, tokenizer, acc_answers, device, max_new_tokens=100)
        for (matched, gen_text), question in zip(acc_results, acc_questions):
            metrics['accuracy'].append(matched)
            metrics['accuracy_predictions'].append({
                "question": question,
                "generated_text": gen_text,
                "answer": edit_item.get(new_answer_key, '')
            })

    return metrics

def edit_mello(model, task_prompt, stop, tokenizer, edit_item, hparams, contriever, contriever_tokenizer, embs, test_generation=False):
    metrics = {'post':
               {        
                'hop_wise':[],
                'accuracy':[],
                }
    }
    device = f"cuda:{hparams.device}"
    t_p = task_prompt
    for i in edit_item['new_single_hops']:
        temp_metrics = []
        ans = i['answer_alias']
        ans.append(i['answer'])
        result, _ = mello(t_p, i['cloze'], model, tokenizer, stop, contriever, contriever_tokenizer, embs, ans, device)
        temp_metrics.append(result)
        result, _ = mello(t_p, i['question'], model, tokenizer, stop, contriever, contriever_tokenizer, embs, ans, device)
        temp_metrics.append(result)
        metrics['hop_wise'].append(temp_metrics)
    answer = edit_item['new_answer_alias']
    answer.append(edit_item['new_answer'])
    res, _ = mello(task_prompt, edit_item['cloze_question'], model, tokenizer, stop, contriever, contriever_tokenizer, embs, answer, device)
    metrics['accuracy'].append(res)
    res, _ = mello(task_prompt, edit_item['questions'][0], model, tokenizer, stop, contriever, contriever_tokenizer, embs, answer, device)
    metrics['accuracy'].append(res)
    return metrics

def cake(original_model, tokenizer, item, hparams, test_generation=False):
    """
    Caveat: return the base model after editing
    """
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"] 
    model = create_lora_model(original_model, target_modules=target_modules)
    # original_model = original_model.to(device)
    model.enable_input_require_grads()
    train_examples = []
    item_case_examples = []
    learning_examples = []
    for rewrite in item['requested_rewrite']:
        """
        prompt: "Fernando Santos is a citizen of"
        target: "United Kingdom"
        
        rewrite_item['question']: "Which country issues Fernando Santos' passport?"
        rewrite_item['answer']: "The United Kingdom issues Fernando Santos' passport as he is a citizen of the United Kingdom."
        
        learning_item['question']: "If Maria lives in a mansion, Fernando Santos resides in a townhouse, and Jose lives in a bungalow, then the person who resides in a townhouse is a citizen of which country?"
        learning_item['answer']: "The person who resides in a townhouse is a citizen of the United Kingdom."
        """
        prompt = rewrite['prompt'].format(rewrite['subject'])
        target = rewrite['target_new']['str']
        item_case_examples.append({
            "text": prompt,
            "target": target
        })
        if 'rephrase_prompt' in rewrite:
            for rewrite_item in rewrite['rephrase_prompt']:
                item_case_examples.append({
                    "text": rewrite_item['question'],
                    "target": rewrite_item['answer']
                })
        if 'learning_prompt' in rewrite:
            for learning_item in rewrite['learning_prompt']:
                learning_examples.append({
                    "text": learning_item['question'],
                    "target": learning_item['answer']
                })
    train_examples.append({'item_case_examples': item_case_examples, 'learning_examples': learning_examples})
    train_dataset = Dataset.from_list(train_examples) # converts a Python list of dicts into Dataset where each list element becomes one row and dict keys become columns
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    training_args = TrainingArguments(
        output_dir=f'./output/',
        overwrite_output_dir=True,
        num_train_epochs=40,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        save_strategy="no",
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    start = time()
    trainer.train()
    exec_time = time() - start
    metrics = {
        'case_id': item['case_id'],
        "requested_rewrite": item['requested_rewrite'],
        "time": exec_time,
        "post": compute_edit_quality(model, tokenizer, item, hparams, test_generation=test_generation),
    }
    model = model.unload()
    del model.peft_config

    return model, metrics


def edit(model, tokenizer, edit_item, hparams,alg_name,apply_algo,test_generation=False):
    # all_metrics = []
    start = time()
    requests = edit_item['requested_rewrite']
    for i in requests:
        if isinstance(i['target_new'], dict):
            i['target_new'] = i['target_new']['str']
    hparams.batch_size = len(requests)
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams,
        copy=False,
        return_orig_weights=True
    )
    exec_time = time() - start
    # start = time()
    # chunk_metrics = []
    metrics = {
        'case_id': edit_item['case_id'],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(edited_model, tokenizer, edit_item, hparams, test_generation=test_generation),
    }
    # chunk_metrics.append(metrics)
    if alg_name == 'KN' or alg_name == 'GRACE' or alg_name == 'WISE':
        with torch.no_grad():
            weights_copy()
    elif alg_name == 'LoRA' or alg_name == 'QLoRA' or alg_name == 'DPO':
        edited_model=edited_model.unload()
        del edited_model.peft_config
    elif alg_name == 'MELO':
        model = edited_model
    else:
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")
    return edited_model, metrics
def edit_rome(model, tokenizer, edit_item, hparams, apply_algo,test_generation=False):
    # all_metrics = []
    start = time()
    requests = edit_item['requested_rewrite']
    print(requests)
    for i in requests:
        if isinstance(i['target_new'], dict):
            i['target_new'] = i['target_new']['str']
    origin_weights_copy = None
    for index,request in enumerate(requests):
        edited_model, weights_copy = apply_algo(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True
        )
        if index == 0:
            origin_weights_copy = weights_copy
    exec_time = time() - start
    metrics = {
        'case_id': edit_item['case_id'],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(edited_model, tokenizer, edit_item, hparams, test_generation=test_generation),
    } 
    with torch.no_grad():
        for k, v in origin_weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")
    return edited_model, metrics
def edit_ifmet(model, tokenizer, edit_item, hparams_s, hparams_d, apply_algo,test_generation=False):
    start = time()
    requests = edit_item['requested_rewrite']
    ifmet_requests = []
    for i in requests:
        if isinstance(i['target_new'], dict):
            i['target_new'] = i['target_new']['str']
        if len(i['ifmet_question'])!=0:
            j={}
            j['prompt'] = i['ifmet_question'][0]
            j['target_new'] = i['target_new']
            j['subject'] = j['prompt']
            j['question'] = i['question']
            ifmet_requests.append(j)
    hparams_s.batch_size = len(requests)
    hparams_d.batch_size = len(ifmet_requests)
    origin_weights_copy = None
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams_s,
        copy=False,
        return_orig_weights=True
    )
    origin_weights_copy = weights_copy
    if len(ifmet_requests) > 0:
        edited_model, weights_copy = apply_algo(
            edited_model,
            tokenizer,
            ifmet_requests,
            hparams_d,
            copy=False,
            return_orig_weights=True
        )
    exec_time = time() - start
    metrics = {
        'case_id': edit_item['case_id'],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(edited_model, tokenizer, edit_item, hparams_s, test_generation=test_generation),
    }
    with torch.no_grad():
        if len(ifmet_requests) > 0:
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams_s.device}")
        for k, v in origin_weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams_s.device}")
    return edited_model, metrics

def edit_wise(model, tokenizer, edit_item, hparams, loc_data, loc_index, apply_algo,test_generation=False):
    start = time()
    requests = edit_item['requested_rewrite']
    for i,item in enumerate(requests):
        item['prompt'] = item['prompt'].format(item['subject'])
        item['target_new'] = item['target_new']['str']
        item.update({
            'loc_prompt': loc_data[loc_index + i]['loc'] + ' ' + loc_data[loc_index + i]['loc_ans']
        })
    loc_index = loc_index + len(requests)
    hparams.batch_size = len(requests)
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams,
        copy=False,
        return_orig_weights=True
    )
    exec_time = time() - start
    metrics = {
        'case_id': edit_item['case_id'],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(edited_model, tokenizer, edit_item, hparams, test_generation=test_generation),
    }
    with torch.no_grad():
        weights_copy()
    return metrics, loc_index


# ============================================================
# Multilingual versions: edit in English, evaluate in 5 languages
# These functions preserve the original functions above unchanged.
# ============================================================

MULTILINGUAL_LANGS = ['en', 'de', 'fr', 'ja', 'zh']

def _eval_all_langs(model, tokenizer, edit_item, hparams, exec_time, test_generation=False):
    """Evaluate edited model in all 5 languages. Returns dict of {lang: metrics}."""
    lang_results = {}
    for lang in MULTILINGUAL_LANGS:
        hparams.lang = lang
        lang_results[lang] = {
            'case_id': edit_item['case_id'],
            'requested_rewrite': edit_item['requested_rewrite'],
            'time': exec_time,
            'post': compute_edit_quality(model, tokenizer, edit_item, hparams, test_generation=test_generation),
        }
    return lang_results

def edit_multilingual(model, tokenizer, edit_item, hparams, alg_name, apply_algo, test_generation=False):
    """Edit in English, evaluate in 5 languages, then restore model."""
    start = time()
    import copy
    requests = copy.deepcopy(edit_item['requested_rewrite'])
    for i in requests:
        if isinstance(i['target_new'], dict):
            i['target_new'] = i['target_new']['str']
    hparams.batch_size = len(requests)
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams,
        copy=False,
        return_orig_weights=True
    )
    exec_time = time() - start

    # Evaluate in all 5 languages on edited model
    lang_results = _eval_all_langs(edited_model, tokenizer, edit_item, hparams, exec_time, test_generation)

    # Restore model
    if alg_name == 'KN' or alg_name == 'GRACE' or alg_name == 'WISE':
        with torch.no_grad():
            weights_copy()
    elif alg_name == 'LoRA' or alg_name == 'QLoRA' or alg_name == 'DPO':
        edited_model = edited_model.unload()
        del edited_model.peft_config
    elif alg_name == 'MELO':
        model = edited_model
    else:
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")
    return edited_model, lang_results

def edit_rome_multilingual(model, tokenizer, edit_item, hparams, apply_algo, test_generation=False):
    """ROME: edit in English, evaluate in 5 languages, then restore model."""
    start = time()
    import copy
    requests = copy.deepcopy(edit_item['requested_rewrite'])
    for i in requests:
        if isinstance(i['target_new'], dict):
            i['target_new'] = i['target_new']['str']
    origin_weights_copy = None
    for index, request in enumerate(requests):
        edited_model, weights_copy = apply_algo(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True
        )
        if index == 0:
            origin_weights_copy = weights_copy
    exec_time = time() - start

    # Evaluate in all 5 languages on edited model
    lang_results = _eval_all_langs(edited_model, tokenizer, edit_item, hparams, exec_time, test_generation)

    # Restore model
    with torch.no_grad():
        for k, v in origin_weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")
    return edited_model, lang_results

def cake_multilingual(original_model, tokenizer, item, hparams, test_generation=False):
    """CaKE: edit in English, evaluate in 5 languages, then restore model."""
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    model = create_lora_model(original_model, target_modules=target_modules)
    model.enable_input_require_grads()
    train_examples = []
    item_case_examples = []
    learning_examples = []
    for rewrite in item['requested_rewrite']:
        prompt = rewrite['prompt'].format(rewrite['subject'])
        target = rewrite['target_new']['str']
        item_case_examples.append({"text": prompt, "target": target})
        if 'rephrase_prompt' in rewrite:
            for rewrite_item in rewrite['rephrase_prompt']:
                item_case_examples.append({"text": rewrite_item['question'], "target": rewrite_item['answer']})
        if 'learning_prompt' in rewrite:
            for learning_item in rewrite['learning_prompt']:
                learning_examples.append({"text": learning_item['question'], "target": learning_item['answer']})
    train_examples.append({'item_case_examples': item_case_examples, 'learning_examples': learning_examples})
    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        preprocess_function, batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )
    training_args = TrainingArguments(
        output_dir=f'./output/', overwrite_output_dir=True,
        num_train_epochs=40, per_device_train_batch_size=4,
        learning_rate=1e-4, save_strategy="no",
        bf16=True, logging_steps=10, report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    start = time()
    trainer.train()
    exec_time = time() - start

    # Evaluate in all 5 languages on edited model
    lang_results = _eval_all_langs(model, tokenizer, item, hparams, exec_time, test_generation)

    # Restore model (unload LoRA)
    model = model.unload()
    del model.peft_config
    return model, lang_results

def edit_wise_multilingual(model, tokenizer, edit_item, hparams, loc_data, loc_index, apply_algo, test_generation=False):
    """WISE: edit in English, evaluate in 5 languages, then restore model."""
    start = time()
    import copy
    requests = copy.deepcopy(edit_item['requested_rewrite'])
    for i, item in enumerate(requests):
        item['prompt'] = item['prompt'].format(item['subject'])
        item['target_new'] = item['target_new']['str']
        item.update({
            'loc_prompt': loc_data[loc_index + i]['loc'] + ' ' + loc_data[loc_index + i]['loc_ans']
        })
    loc_index = loc_index + len(requests)
    hparams.batch_size = len(requests)
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams,
        copy=False,
        return_orig_weights=True
    )
    exec_time = time() - start

    # Evaluate in all 5 languages on edited model
    lang_results = _eval_all_langs(edited_model, tokenizer, edit_item, hparams, exec_time, test_generation)

    # Restore model
    with torch.no_grad():
        weights_copy()
    return lang_results, loc_index
