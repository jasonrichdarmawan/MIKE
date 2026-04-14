from httpcore import request
import torch
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
import random
from datasets import Dataset
from transformers import TrainingArguments, Trainer

from tqdm.auto import tqdm
from typing import Literal

def _extract_answer_text(prediction):
    """Extract answer from <answer> tags, or return raw prediction if no tags found."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", prediction, re.IGNORECASE | re.DOTALL)
    text = m.group(1) if m else prediction
    return text.strip()

def check_answer_in_pred(pred, answers, extract_answer: Literal[False, "answer_tags"] = False):
    if extract_answer == "answer_tags":
        pred = _extract_answer_text(pred).lower()
    else:
        pred = pred.lower()
    for a in answers:
        a = a.strip().lower()
        if not a:
            continue
        if a in pred or pred.startswith(a) or pred == a:
            return True
    return False

def _format_chat_prompt(tokenizer, user_content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Wrap your final answer in XML tags like this: <answer>ANSWER_HERE</answer>."},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _format_chat_train_pair(tokenizer, text, target, apply_chat_template=True):
    """
    Example:
    - text: ""If Maria lives in a mansion, Fernando Santos resides in a townhouse, and Jose lives in a bungalow, then the person who resides in a townhouse is a citizen of which country?"
    - target: "The person who resides in a townhouse is a citizen of the United Kingdom."
    """
    if not apply_chat_template:
        full_text = f"{text} {target}"
        prompt_text = text
        return full_text, prompt_text

    full_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": target},
    ]
    prompt_messages = [{"role": "user", "content": text}]

    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return full_text, prompt_text

def get_sentence_embeddings(
    contriever, 
    contriever_tokenizer, 
    sentences, 
    batch_size=32
):
    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing sentence embeddings in batches"):
        sent_batch = sentences[i:i+batch_size]
        contriever_tokenizer.padding_side = "right"
        inputs = contriever_tokenizer(
            sent_batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(contriever.device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings)
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts_batch(
    contriever,
    contriever_tokenizer,
    queries,
    facts_embeddings,
    k=1,
    batch_size=32,
):
    if len(queries) == 0:
        raise ValueError("queries is empty")

    contriever_tokenizer.padding_side = "right"
    all_topk = []

    for start in tqdm(range(0, len(queries), batch_size), desc="Retrieving facts in batches"):
        end = min(start + batch_size, len(queries))
        query_batch = queries[start:end]

        inputs = contriever_tokenizer(
            query_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(contriever.device)

        with torch.no_grad():
            outputs = contriever(**inputs)
            query_emb = mean_pooling(outputs[0], inputs['attention_mask'])

        sim = query_emb @ facts_embeddings.T
        topk = sim.topk(k, largest=True).indices
        all_topk.append(topk)

    all_topk = torch.cat(all_topk, dim=0)
    if k == 1:
        return all_topk[:, 0].tolist()
    return all_topk.tolist()

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.) # shape is (batch_size, seq_len, hidden_dim)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def create_lora_model(
    model,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] = None,
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

few_shot_in_multi_turn = [
    {"role": "system", "content":
"""You are a helpful assistant. Answer in the same language as the question. You must answer questions by strictly following this step-by-step pattern:
Question: [Original question]
Subquestion: [Your generated subquestion to retrieve a fact that can help answer the original question]
Generated answer: [Your generated answer to the subquestion]
Retrieved fact: [The new fact retrieved from a knowledge base based on the generated subquestion]
Retrieved fact {contradicts/does not contradict} to generated answer, so the intermediate answer is: [Your revised answer based on the retrieved fact]
(Repeat the subquestion cycle as needed)
Final answer: [Definitive conclusion]"""},
    {"role": "user", "content": "Question: What is the capital city of the country of citizenship of Ivanka Trump's spouse?"},
    {"role": "assistant", "content":
"""Subquestion: Who is Ivanka Trump's spouse?
Generated answer: Ivanka Trump's spouse is Jared Kushner.
Retrieved fact: Ivanka Biden is married to Samantha Cameron.
Retrieved fact does not contradict to generated answer, so the intermediate answer is: Jared Kushner
Subquestion: What is the country of citizenship of Jared Kushner?
Generated answer: The country of citizenship of Jared Kushner is United States.
Retrieved fact: Jared Kushner is a citizen of Canada.
Retrieved fact contradicts to generated answer, so the intermediate answer is: Canada
Subquestion: What is the capital city of Canada?
Generated answer:  The capital city of Canada is Ottawa.
Retrieved fact: The capital city of United States of America is Seattle.
Retrieved fact does not contradict to generated answer, so the intermediate answer is: Ottawa
Final answer: Ottawa"""},
    {"role": "user", "content": "Question: Who is the head of state of the country where Rainn Wilson holds a citizenship?"},
    {"role": "assistant", "content":
"""Subquestion: What is the country of citizenship of Rainn Wilson?
Generated answer: Rainn Wilson is a citizen of United States of America.
Retrieved fact: Rainn Wilson is a citizen of Croatia.
Retrieved fact contradicts to generated answer, so the intermediate answer is: Croatia
Subquestion: What is the name of the current head of state in Croatia?
Generated answer: The head of state of Croatia is President Zoran Milanović.
Retrieved fact: The name of the current head of state in Croatia is Kolinda Grabar-Kitarović.
Retrieved fact contradicts to generated answer, so the intermediate answer is: Kolinda Grabar-Kitarović
Final answer: Kolinda Grabar-Kitarović"""},
    {"role": "user", "content": "Question: Who is the spouse of the head of state in United States of America?"},
    {"role": "assistant", "content":
"""Subquestion: Who is the head of state in United States of America?
Generated answer: The US president is Donald Trump.
Retrieved fact: The head of state in United States of America is Joe Biden.
Retrieved fact contradicts to generated answer, so the intermediate answer is: Joe Biden
Subquestion: Who is the spouse of Joe Biden?
Generated answer: The spouse of Joe Biden is Jill Biden.
Retrieved fact: The spouse of Joe Bill is Evan Austin.
Retrieved fact does not contradict to generated answer, so the intermediate answer is: Jill Biden
Final answer: Jill Biden"""},
    {"role": "user", "content": "Question: On which continent is the country of citizenship of the founder of the manufacturer of iPhone 5 situated?"},
    {"role": "user", "content":
"""Subquestion: Which company is iPhone 5 produced by?
Generated answer: iPhone 5 is produced by Apple.
Retrieved fact: The company that produced iPhone 5 is Iveco.
Retrieved fact contradicts to generated answer, so the intermediate answer is: Iveco
Subquestion: Who is the founder of Iveco?
Generated answer: Iveco was founded by Giovanni Agnelli.
Retrieved fact: House of Bonaparte was founded by Gustav I of Sweden.
Retrieved fact does not contradict to generated answer, so the intermediate answer is: Giovanni Agnelli
Subquestion: What is the country of citizenship of Giovanni Agnelli?
Generated answer: Giovanni Agnelli is a citizen of Italy.
Retrieved fact: Giovanni Agnelli is a citizen of Niger.
Retrieved fact contradicts to generated answer, so the intermediate answer is: Niger.
Subquestion: On which continent is Niger situated?
Generated answer: Niger is situated on Africa.
Retrieved fact: Kingdom of England is located in the continent of North America.
Retrieved fact does not contradict to generated answer, so the intermediate answer is: Africa.
Final answer: Africa"""},
]

def call_model_batch(
    model,
    tokenizer,
    user_contents,
    stop_strings,
    eval_with_chat_template: Literal[False, "few_shot_in_multi_turn", "few_shot_in_user_content"] = False,
    max_new_tokens=256,
    batch_size=8,
    continuation: Literal["in_user_content", "in_assistant_content"] = "in_user_content",
    assistant_histories=None,
):
    if len(user_contents) == 0:
        raise ValueError("user_contents is empty")

    generated_all = []

    if continuation == "in_user_content":
        formatted_prompts = _format_mello_prompts(
            tokenizer=tokenizer,
            prompts=user_contents,
            eval_with_chat_template=eval_with_chat_template,
        )
    elif continuation == "in_assistant_content":
        if assistant_histories is None or len(assistant_histories) != len(user_contents):
            raise ValueError("assistant_histories must match user_contents length for in_assistant_content")

        formatted_prompts = []
        for question, assistant_history in zip(user_contents, assistant_histories):
            if eval_with_chat_template == "few_shot_in_multi_turn":
                messages = few_shot_in_multi_turn + [{"role": "user", "content": question}]
            elif eval_with_chat_template == "few_shot_in_user_content":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ]
            else:
                raise ValueError(f"Invalid eval_with_chat_template value: {eval_with_chat_template}")

            if assistant_history:
                messages.append({"role": "assistant", "content": assistant_history})
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_prompts.append(prompt)
    else:
        raise ValueError(f"Invalid continuation value: {continuation}")

    for start in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating responses in batches"):
        end = min(start + batch_size, len(formatted_prompts))
        batch_prompts = formatted_prompts[start:end]

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=not eval_with_chat_template,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stop_strings=stop_strings,
                tokenizer=tokenizer,
                use_cache=True,
            )

        new_token_ids = outputs[:, inputs.input_ids.shape[1]:]
        generated_batch = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for generated_text in generated_batch:
            generated_text = _trim_generated_text(generated_text, stop_strings)
            generated_all.append(generated_text)

    return generated_all

def preprocess_function(examples, tokenizer, apply_chat_template=False, mask_prompt_part=False):
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
        full_text, prompt_text = _format_chat_train_pair(tokenizer, text, target, apply_chat_template=apply_chat_template)
        full_sequences.append(full_text)
        prompt_sequences.append(prompt_text)

    model_inputs = tokenizer(
        full_sequences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np",
        add_special_tokens=not apply_chat_template,
    )
    prompt_inputs = tokenizer(
        prompt_sequences,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors=None,
        add_special_tokens=not apply_chat_template,
    )

    labels = model_inputs["input_ids"].copy()
    if mask_prompt_part:
        for idx, prompt_ids in enumerate(prompt_inputs["input_ids"]):
            prefix_length = min(len(prompt_ids), labels.shape[1])
            labels[idx, :prefix_length] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
    model_inputs['labels'] = labels
    return model_inputs

def check_answer_batch(
    model,
    tokenizer,
    questions,
    answers,
    eval_with_chat_template=False,
    max_new_tokens=50,
    batch_size=8,
):
    """
    Batch version of `check_answer`.

    Args:
    - questions: List[str]
    - answers: List[List[str]]

    Returns:
    - List[Tuple[prompt, generated_text, is_correct]]
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if len(questions) == 0:
        return []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = []

    try:
        formatted_prompts = questions
        if eval_with_chat_template:
            formatted_prompts = [
                _format_chat_prompt(tokenizer=tokenizer, user_content=question)
                for question in questions
            ]

        for start in range(0, len(formatted_prompts), batch_size):
            end = min(start + batch_size, len(formatted_prompts))
            batch_prompts = formatted_prompts[start:end]
            batch_answers = answers[start:end]

            tokens = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not eval_with_chat_template,
            ).to(model.device)

            input_pad_length = tokens["input_ids"].shape[1]
            outputs = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
            generated_ids_batch = outputs[:, input_pad_length:]
            generated_texts = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
            
            for prompt_text, generated_text, expected_answers in zip(batch_prompts, generated_texts, batch_answers):
                results.append(
                    (
                        prompt_text,
                        generated_text,
                        check_answer_in_pred(generated_text, expected_answers),
                    )
                )

    finally:
        tokenizer.padding_side = original_padding_side

    return results

def _format_mello_prompts(tokenizer, prompts, eval_with_chat_template):
    if eval_with_chat_template == "few_shot_in_user_content":
        return [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    elif eval_with_chat_template == "few_shot_in_multi_turn":
        return [
            tokenizer.apply_chat_template(
                few_shot_in_multi_turn + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    elif eval_with_chat_template is False:
        return prompts
    else:
        raise ValueError(f"Invalid eval_with_chat_template value: {eval_with_chat_template}")

def _trim_generated_text(generated_text, stop_strings):
    for stop_word in stop_strings:
        if stop_word in generated_text:
            return generated_text.split(stop_word)[0].strip()
    return generated_text

def _select_past_key_values(past_key_values, batch_indices):
    return tuple(
        tuple(state.index_select(0, batch_indices) for state in layer)
        for layer in past_key_values
    )

def _get_past_length(past_key_values):
    return past_key_values[0][0].shape[2]

def _build_prefix_attention_mask(
    sequences,
    pad_token_id,
    trailing_trim_token_ids,
    trailing_trim_token_sequences=[],
):
    prefix_mask = (sequences != pad_token_id).long()
    for row in range(sequences.shape[0]):
        last_idx = sequences.shape[1] - 1
        while last_idx >= 0 and prefix_mask[row, last_idx] == 0:
            last_idx -= 1
        while last_idx >= 0 and int(sequences[row, last_idx].item()) in trailing_trim_token_ids:
            prefix_mask[row, last_idx] = 0
            last_idx -= 1

        for token_seq in trailing_trim_token_sequences:
            seq_len = len(token_seq)
            if seq_len == 0:
                continue
            while last_idx - seq_len + 1 >= 0:
                start_idx = last_idx - seq_len + 1
                if torch.all(sequences[row, start_idx:last_idx + 1] == token_seq):
                    prefix_mask[row, start_idx:last_idx + 1] = 0
                    last_idx = start_idx - 1
                else:
                    break
    return prefix_mask


def _evaluate_mello_with_kv_cache(
    model,
    tokenizer,
    contriever,
    contriever_tokenizer,
    task_prompt,
    items,
    embeddings,
    new_facts,
    eval_with_chat_template: Literal[False, "few_shot_in_multi_turn", "few_shot_in_user_content"] = False,
    generation_batch_size=8,
    languages=["en", "de", "fr", "ja", "zh"],
):
    stop_strings = ["Retrieved fact:"]
    
    all_metrics, eval_requests = _prepare_mquake_clke(items, hparams=None, languages=languages)
    
    request_outputs = []

    if len(eval_requests) == 0:
        raise ValueError("No evaluation requests prepared for MeLLo evaluation.")

    trailing_trim_token_ids = set()
    if tokenizer.eos_token_id is not None:
        trailing_trim_token_ids.add(tokenizer.eos_token_id)
    for maybe_end_token in ("<|im_end|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(maybe_end_token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            trailing_trim_token_ids.add(token_id)

    trailing_trim_token_sequences = []
    for stop_text in stop_strings:
        for variant in (stop_text, f"\n{stop_text}", f" {stop_text}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) > 0:
                trailing_trim_token_sequences.append(
                    torch.tensor(token_ids, dtype=torch.long, device=model.device)
                )

    for chunk_start in tqdm(range(0, len(eval_requests), generation_batch_size), desc="Evaluating MeLLo with KV cache"):
        chunk_end = min(chunk_start + generation_batch_size, len(eval_requests))
        chunk_requests = eval_requests[chunk_start:chunk_end]
        chunk_size = len(chunk_requests)
        
        initial_prompts = []
        for req in chunk_requests:
            if eval_with_chat_template == "few_shot_in_multi_turn":
                init_prompt = f"Question: {req['question']}"
            elif eval_with_chat_template == "few_shot_in_user_content":
                init_prompt = f"{task_prompt}\n\nQuestion: {req['question']}"
            else:
                raise ValueError(f"Invalid eval_with_chat_template value: {eval_with_chat_template}")
            initial_prompts.append(init_prompt)

        formatted_prompts = _format_mello_prompts(
            tokenizer=tokenizer, 
            prompts=initial_prompts, 
            eval_with_chat_template=eval_with_chat_template
        )
        
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=not eval_with_chat_template,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                stop_strings=stop_strings,
                tokenizer=tokenizer,
                return_dict_in_generate=True,
                use_cache=True,
            )

        step_new_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
        step_texts = [
            _trim_generated_text(text, stop_strings)
            for text in tokenizer.batch_decode(step_new_ids, skip_special_tokens=True)
        ]

        histories = [""] * chunk_size
        local_outputs = [None] * chunk_size
        active_local_indices = list(range(chunk_size))
        active_past = out.past_key_values
        active_sequences = out.sequences

        for step_idx in tqdm(range(4), desc="Evaluating MeLLo in iterative generation"):
            continue_batch_positions = []
            continue_local_indices = []
            continue_subquestions = []

            for batch_pos, local_idx in enumerate(active_local_indices):
                gen = step_texts[batch_pos]
                histories[local_idx] += gen
                req = chunk_requests[local_idx]

                last_sent = gen.strip().split('\n')[-1]
                if last_sent.startswith('Final answer: '):
                    answer = last_sent[len("Final answer: "):]
                    local_outputs[local_idx] = (
                        req['question'],
                        histories[local_idx],
                        check_answer_in_pred(answer, req['answers']),
                    )
                    continue

                lines = gen.strip().split('\n')
                if len(lines) < 2:
                    local_outputs[local_idx] = (req['question'], histories[local_idx], False)
                    continue

                subquestion = lines[-2]
                if not subquestion.startswith('Subquestion: '):
                    local_outputs[local_idx] = (req['question'], histories[local_idx], False)
                    continue

                subquestion = subquestion[len("Subquestion: "):]
                continue_batch_positions.append(batch_pos)
                continue_local_indices.append(local_idx)
                continue_subquestions.append(subquestion)

            if len(continue_local_indices) == 0 or step_idx == 3:
                break

            top_fact_indices = retrieve_facts_batch(
                contriever=contriever,
                contriever_tokenizer=contriever_tokenizer,
                queries=continue_subquestions,
                facts_embeddings=embeddings,
                k=1,
                batch_size=generation_batch_size,
            )

            fact_texts = []
            for fact_idx, local_idx in zip(top_fact_indices, continue_local_indices):
                fact_sent = new_facts[fact_idx]
                fact_text = f"\nRetrieved fact: {fact_sent}."
                histories[local_idx] += fact_text
                fact_texts.append(fact_text)

            keep_idx = torch.tensor(continue_batch_positions, device=model.device)
            selected_past = _select_past_key_values(past_key_values=active_past, batch_indices=keep_idx)
            selected_sequences = active_sequences.index_select(0, keep_idx)

            tokenizer.padding_side = "left"
            fact_tokens = tokenizer(
                fact_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)

            fact_input_ids = fact_tokens["input_ids"]
            fact_attention_mask = fact_tokens["attention_mask"]

            prefix_attention_mask = _build_prefix_attention_mask(
                selected_sequences,
                tokenizer.pad_token_id,
                trailing_trim_token_ids,
                trailing_trim_token_sequences=trailing_trim_token_sequences,
            )
            attention_mask = torch.cat([prefix_attention_mask, fact_attention_mask], dim=1)
            past_length = _get_past_length(selected_past)
            cache_position = torch.arange(
                past_length,
                past_length + fact_input_ids.shape[1],
                dtype=torch.int64,
                device=model.device,
            )

            with torch.no_grad():
                out = model.generate(
                    input_ids=fact_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=selected_past,
                    cache_position=cache_position,
                    max_new_tokens=256,
                    do_sample=False,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

            step_new_ids = out.sequences[:, fact_input_ids.shape[1]:]
            step_texts = [
                _trim_generated_text(text, stop_strings)
                for text in tokenizer.batch_decode(step_new_ids, skip_special_tokens=True)
            ]

            active_local_indices = continue_local_indices
            active_past = out.past_key_values
            active_sequences = torch.cat([selected_sequences, out.sequences], dim=1)

        for local_idx, output in enumerate(local_outputs):
            if output is None:
                req = chunk_requests[local_idx]
                local_outputs[local_idx] = (req['question'], histories[local_idx], False)

        for local_idx, output in enumerate(local_outputs):
            global_req_idx = chunk_start + local_idx
            request_outputs.append((global_req_idx, output))

    output_map = {req_idx: output for req_idx, output in request_outputs}
    for req_idx, req in enumerate(eval_requests):
        question, generated, result = output_map[req_idx]
        item_post = all_metrics[req["language"]][req['item_idx']]['post']
        prediction = {
            "prompt": question,
            "generated": generated,
            "target": req['target'],
        }

        if req['group'] == 'hop_wise':
            item_post['hop_wise'][req['hop_idx']][req['form_idx']] = result
            item_post['hop_wise_predictions'][req['hop_idx']][req['form_idx']] = prediction
        else:
            item_post['accuracy'].append(result)
            item_post['accuracy_predictions'].append(prediction)

    return all_metrics

def _evaluate_mello_without_kv_cache(
    model,
    tokenizer,
    contriever,
    contriever_tokenizer,
    task_prompt,
    items,
    embeddings,
    new_facts,
    eval_with_chat_template: Literal[False, "few_shot_in_multi_turn", "few_shot_in_user_content"] = False,
    generation_batch_size=8,
    languages=["en", "de", "fr", "ja", "zh"],
    continuation: Literal["in_user_content", "in_assistant_content"] = "in_user_content",
):
    stop_strings = ["Retrieved fact:"]
    
    all_metrics, eval_requests = _prepare_mquake_clke(items, hparams=None, languages=languages)
    
    request_outputs = []
    initial_prompts = []
    prompts = []
    assistant_histories = []
    done = []
    last_gen = []

    for req in eval_requests:
        if eval_with_chat_template == "few_shot_in_multi_turn":
            init_prompt = f"Question: {req['question']}"
        elif eval_with_chat_template == "few_shot_in_user_content":
            init_prompt = f"{task_prompt}\n\nQuestion: {req['question']}"
        else:
            raise ValueError(f"Invalid eval_with_chat_template value: {eval_with_chat_template}")

        initial_prompts.append(init_prompt)
        prompts.append(init_prompt)
        assistant_histories.append("")
        request_outputs.append(None)
        done.append(False)
        last_gen.append("")

    for _ in tqdm(range(4), desc="Evaluating MeLLo in iterative generation"):
        active_indices = [idx for idx, is_done in enumerate(done) if not is_done]
        if len(active_indices) == 0:
            break

        if continuation == "in_user_content":
            active_prompts = [prompts[idx] for idx in active_indices]
            generated_batch = call_model_batch(
                model=model,
                tokenizer=tokenizer,
                user_contents=active_prompts,
                stop_strings=stop_strings,
                eval_with_chat_template=eval_with_chat_template,
                batch_size=generation_batch_size,
                continuation=continuation,
            )
        elif continuation == "in_assistant_content":
            active_questions = [eval_requests[idx]['question'] for idx in active_indices]
            active_assistant_histories = [assistant_histories[idx] for idx in active_indices]
            generated_batch = call_model_batch(
                model=model,
                tokenizer=tokenizer,
                user_contents=active_questions,
                stop_strings=stop_strings,
                eval_with_chat_template=eval_with_chat_template,
                batch_size=generation_batch_size,
                continuation=continuation,
                assistant_histories=active_assistant_histories,
            )
        else:
            raise ValueError(f"Invalid continuation value: {continuation}")

        retrieval_candidates = []

        for active_pos, req_idx in enumerate(active_indices):
            gen = generated_batch[active_pos]
            last_gen[req_idx] = gen
            req = eval_requests[req_idx]

            if continuation == "in_assistant_content":
                assistant_histories[req_idx] = assistant_histories[req_idx] + gen

            last_sent = gen.strip().split('\n')[-1]
            if last_sent.startswith('Final answer: '):
                answer = last_sent[len("Final answer: "):]
                generated = (
                    assistant_histories[req_idx]
                    if continuation == "in_assistant_content"
                    else (prompts[req_idx] + gen)[len(initial_prompts[req_idx]):]
                )
                request_outputs[req_idx] = (
                    req['question'],
                    generated,
                    check_answer_in_pred(answer, req['answers']),
                )
                done[req_idx] = True
                continue

            lines = gen.strip().split('\n')
            if len(lines) < 2:
                generated = (
                    assistant_histories[req_idx]
                    if continuation == "in_assistant_content"
                    else (prompts[req_idx] + gen)[len(initial_prompts[req_idx]):]
                )
                request_outputs[req_idx] = (req['question'], generated, False)
                done[req_idx] = True
                continue

            subquestion = lines[-2]
            if not subquestion.startswith('Subquestion: '):
                generated = (
                    assistant_histories[req_idx]
                    if continuation == "in_assistant_content"
                    else (prompts[req_idx])[len(initial_prompts[req_idx]):]
                )
                request_outputs[req_idx] = (req['question'], generated, False)
                done[req_idx] = True
                continue

            subquestion = subquestion[len("Subquestion: "):]
            retrieval_candidates.append((req_idx, gen, subquestion))

        if len(retrieval_candidates) > 0:
            subquestions = [candidate[2] for candidate in retrieval_candidates]
            top_fact_indices = retrieve_facts_batch(
                contriever=contriever,
                contriever_tokenizer=contriever_tokenizer,
                queries=subquestions,
                facts_embeddings=embeddings,
                k=1,
                batch_size=generation_batch_size,
            )
            for candidate_idx, (req_idx, gen, _) in enumerate(retrieval_candidates):
                fact_sent = new_facts[top_fact_indices[candidate_idx]]
                prompts[req_idx] = f"{prompts[req_idx]}\n{gen}\nRetrieved fact: {fact_sent}."
                if continuation == "in_assistant_content":
                    assistant_histories[req_idx] = assistant_histories[req_idx] + f"\nRetrieved fact: {fact_sent}."

    for req_idx, is_done in enumerate(done):
        if is_done:
            continue
        req = eval_requests[req_idx]
        generated = (
            assistant_histories[req_idx]
            if continuation == "in_assistant_content"
            else (prompts[req_idx])[len(initial_prompts[req_idx]):]
        )
        request_outputs[req_idx] = (req['question'], generated, False)

    for req, output in zip(eval_requests, request_outputs):
        question, generated, result = output
        item_post = all_metrics[req["languages"]][req['item_idx']]['post']
        prediction = {
            "prompt": question,
            "generated": generated,
            "target": req['target'],
        }

        if req['group'] == 'hop_wise':
            item_post['hop_wise'][req['hop_idx']][req['form_idx']] = result
            item_post['hop_wise_predictions'][req['hop_idx']][req['form_idx']] = prediction
        else:
            item_post['accuracy'].append(result)
            item_post['accuracy_predictions'].append(prediction)
    
    return all_metrics


def evaluate_mello(
    model,
    tokenizer,
    contriever,
    contriever_tokenizer,
    task_prompt,
    items,
    embeddings,
    new_facts,
    eval_with_chat_template: Literal[False, "few_shot_in_multi_turn", "few_shot_in_user_content"] = False,
    generation_batch_size=8,
    languages=["en", "de", "fr", "ja", "zh"],
    use_kv_cache=False,
    continuation: Literal["in_user_content", "in_assistant_content"] = "in_user_content",
):
    """
    continuation is only applicable when use_kv_cache=False
    """
    if use_kv_cache:
        return _evaluate_mello_with_kv_cache(
            model=model,
            tokenizer=tokenizer,
            contriever=contriever,
            contriever_tokenizer=contriever_tokenizer,
            task_prompt=task_prompt,
            items=items,
            embeddings=embeddings,
            new_facts=new_facts,
            eval_with_chat_template=eval_with_chat_template,
            generation_batch_size=generation_batch_size,
            languages=languages,
        )
    
    return _evaluate_mello_without_kv_cache(
        model=model,
        tokenizer=tokenizer,
        contriever=contriever,
        contriever_tokenizer=contriever_tokenizer,
        task_prompt=task_prompt,
        items=items,
        embeddings=embeddings,
        new_facts=new_facts,
        eval_with_chat_template=eval_with_chat_template,
        generation_batch_size=generation_batch_size,
        languages=languages,
        continuation=continuation,
    )

def _edit_cake_single(model, tokenizer, item):
    """
    Caveat: return the base model after editing
    """
    if not hasattr(model, "peft_config"):
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"] 
        model = create_lora_model(model, target_modules=target_modules)
    
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
        item_case_examples.append({
            "text": rewrite['prompt'].format(rewrite['subject']),
            "target": rewrite['target_new']['str'],
        })
        if 'rephrase_prompt' in rewrite:
            for rewrite_item in rewrite['rephrase_prompt']:
                item_case_examples.append({
                    "text": rewrite_item['question'],
                    "target": rewrite_item['answer'],
                })
        if 'learning_prompt' in rewrite:
            for learning_item in rewrite['learning_prompt']:
                learning_examples.append({
                    "text": learning_item['question'],
                    "target": learning_item['answer'],
                })
    
    train_examples = []
    train_examples.append({
        'item_case_examples': item_case_examples, 
        'learning_examples': learning_examples,
    })
    train_dataset = Dataset.from_list(train_examples) # converts a Python list of dicts into Dataset where each list element becomes one row and dict keys become columns
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    training_args = TrainingArguments(
        output_dir=f'./output/',
        overwrite_output_dir=False,
        num_train_epochs=40,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        save_strategy="no",
        bf16=True,
        logging_steps=10,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    
    # CaKE train per-sample and so this function is called in a loop, causing RecursionError as Trainer class wrap the model multiple times.
    model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)

    return model

def edit_cake(model, tokenizer, items):
    """
    Supports both single-item and batched editing for CaKE.

    Caveat:
    - CaKE edits one item at a time.
    - If `items` is a list, edits are applied sequentially in list order.
    - Return type matches input style:
      - single dict input -> `(model, metrics_dict)`
      - list input -> `(model, metrics_list)`
    """

    for item in items:
        model = _edit_cake_single(
            model=model,
            tokenizer=tokenizer,
            item=item,
        )

    return model


def edit(model, tokenizer, items, hparams, apply_algo):
    requests = []
    for item in items:
        requested_rewrite = item['requested_rewrite']
        for i in requested_rewrite:
            if isinstance(i["target_true"], dict):
                i['target_true'] = i['target_true']['str']
            if isinstance(i["target_new"], dict):
                i['target_new'] = i['target_new']['str']
        requests.extend(requested_rewrite)
    
    tokenizer.padding_side = "right"
    if hparams.alg_name in ["ROME", "R-ROME"]: # Editing method that edit one sample at a time.
        for request in requests:
            model, _ = apply_algo(
                model,
                tokenizer,
                [request],
                hparams,
                copy=False,
                return_orig_weights=False,
            )
    else:
        model = apply_algo(
            model,
            tokenizer,
            requests,
            hparams,
            copy=False,
            return_orig_weights=False,
        )
    
    return model

def _prepare_mquake_clke(items, hparams, languages):
    """
    dataset: MQuAKE-CF-3k-v2-clke
    """
    all_metrics = {language: [] for language in languages}
    eval_requests = []
    
    for item in items:
        requested_rewrite = item['requested_rewrite']
        for i in requested_rewrite:
            if isinstance(i["target_true"], dict):
                i['target_true'] = i['target_true']['str']
            if isinstance(i["target_new"], dict):
                i['target_new'] = i['target_new']['str']
    
    for item_idx, item in enumerate(tqdm(items, desc="Preparing eval requests")):
        requested_rewrite = []
        if hparams is not None:
            for i in item['requested_rewrite']:
                z = (
                    i
                    if hparams.alg_name == "CaKE"
                    else {k: v for k, v in i.items() if k not in ["rephrase_prompt", "learning_prompt"]}
                )
                requested_rewrite.append(z)

        for language in languages:
            language_str = "" if language == "en" else f"_{language}"
            hop_count = len(item[f'new_single_hops{language_str}'])
            metrics = {
                "case_id": item['case_id'],
                "requested_rewrite": requested_rewrite,
                "post": {
                    "hop_wise": [[None, None] for _ in range(hop_count)],
                    "accuracy": [],
                    "hop_wise_predictions": [[None, None] for _ in range(hop_count)],
                    "accuracy_predictions": [],
                },
            }
            all_metrics[language].append(metrics)

        for language in languages:
            language_str = "" if language == "en" else f"_{language}"
            for hop_idx, single_hop in enumerate(item[f'new_single_hops{language_str}']):
                answer_candidates = [single_hop["answer"]] + single_hop['answer_alias']
                eval_requests.append({
                    "item_idx": item_idx,
                    "language": language,
                    "group": "hop_wise",
                    "hop_idx": hop_idx,
                    "form_idx": 0,
                    "question": single_hop['cloze'],
                    "answers": answer_candidates,
                    "target": single_hop["answer"],
                    "max_new_tokens": 50,
                })
                eval_requests.append({
                    "item_idx": item_idx,
                    "language": language,
                    "group": "hop_wise",
                    "hop_idx": hop_idx,
                    "form_idx": 1,
                    "question": single_hop["question"],
                    "answers": answer_candidates,
                    "target": single_hop["answer"],
                    "max_new_tokens": 50,
                })

        for language in languages:
            language_str = "" if language == "en" else f"_{language}"
            answer_candidates = [item[f"new_answer{language_str}"]] + item[f'new_answer_alias{language_str}']
            eval_requests.append({
                "item_idx": item_idx,
                "language": language,
                "group": "accuracy",
                "question": item[f'cloze_question{language_str}'],
                "answers": answer_candidates,
                "target": item[f"new_answer{language_str}"],
                "max_new_tokens": 50,
            })
            for question in item[f'questions{language_str}']:
                eval_requests.append({
                    "item_idx": item_idx,
                    "language": language,
                    "group": "accuracy",
                    "question": question,
                    "answers": answer_candidates,
                    "target": item[f"new_answer{language_str}"],
                    "max_new_tokens": 50,
                })
                
    return all_metrics, eval_requests

def evaluate(model, tokenizer, items, hparams, eval_with_chat_template, generation_batch_size=8, languages=["en", "de", "fr", "ja", "zh"]):
    """
    Caveat:
    - CaKE's implementation: metrics['accuracy'] only evaluate edit_item['questions'][0]
    
    Example:
    - edit_item['new_answer_alias']: ["Emerson Dambudzo Mnangagwa", ...]
    - edit_item['new_answer']: "Emerson Mnangagwa"
    - edit_item['cloze_question']: "The head of state of the country where Fernando Santos holds citizenship is"
    - edit_item['questions']: ["Who is the head of state of the country where Fernando Santos hold a citizenship?"]
    """
    all_metrics, eval_requests = _prepare_mquake_clke(items=items, hparams=hparams, languages=languages)

    grouped_indices = {}
    for req_idx, req in enumerate(eval_requests):
        grouped_indices.setdefault(req["max_new_tokens"], []).append(req_idx)

    eval_outputs = [None] * len(eval_requests)
    for max_new_tokens, req_indices in tqdm(grouped_indices.items(), desc="Evaluating in batches"):
        batch_questions = [eval_requests[idx]["question"] for idx in req_indices]
        batch_answers = [eval_requests[idx]["answers"] for idx in req_indices]
        batch_results = check_answer_batch(
            model=model,
            tokenizer=tokenizer,
            questions=batch_questions,
            answers=batch_answers,
            eval_with_chat_template=eval_with_chat_template,
            max_new_tokens=max_new_tokens,
            batch_size=generation_batch_size,
        )
        for local_idx, req_idx in enumerate(req_indices):
            eval_outputs[req_idx] = batch_results[local_idx]

    for req, output in zip(eval_requests, eval_outputs):
        prompt, generated_text, result = output
        item_post = all_metrics[req["language"]][req["item_idx"]]["post"]
        prediction = {
            "prompt": prompt,
            "generated": generated_text,
            "target": req["target"],
        }

        if req["group"] == "hop_wise":
            item_post["hop_wise"][req["hop_idx"]][req["form_idx"]] = result
            item_post["hop_wise_predictions"][req["hop_idx"]][req["form_idx"]] = prediction
        elif req["group"] == "accuracy":
            item_post["accuracy"].append(result)
            item_post["accuracy_predictions"].append(prediction)
        else:
            raise ValueError(f"Unknown request group: {req['group']}")

    return all_metrics

def edit_ifmet(
    model,
    tokenizer,
    items,
    hparams_shallow,
    hparams_deeper,
    apply_algo,
):
    """
    Caveat: the [baseline repository](https://github.com/zjunlp/CaKE/tree/df528e13fffbd2117e6041c129285e1e146933d2) evaluate against MQuAKE benchmark. 
    The provided benchmark do not have "ifmet_question" key, so the deeper editing stage is effectively skipped, i.e. the same as MEMIT.
    
    According to the paper, the prompt criteria for the deeper editing stage is a two-hop question in cloze format.
    """
    requests = []
    ifmet_requests = []
    for item in items:
        for i in item["requested_rewrite"]:
            if isinstance(i["target_new"], dict):
                i['target_new'] = i['target_new']['str']
            if item.get("ifmet_question") is not None and isinstance(i["ifmet_question"], list) and len(i["ifmet_question"]) > 0:
                j = {
                    "prompt": i["ifmet_question"][0],
                    "target_new": i["target_new"],
                    "subject": i["ifmet_question"][0],
                    "question": i["question"],
                }
                ifmet_requests.append(j)
        requests.extend(item["requested_rewrite"])
    
        if item.get("cloze_question") is not None:
            j = {
                "prompt": item["cloze_question"],
                "target_new": item["new_answer"],
                "subject": item["cloze_question"],
                "question": item["cloze_question"],
            }
            ifmet_requests.append(j)
    
    hparams_shallow.batch_size = len(requests)
    hparams_deeper.batch_size = len(ifmet_requests)

    model, _ = apply_algo(
        model,
        tokenizer,
        requests,
        hparams_shallow,
        copy=False,
        return_orig_weights=False
    )

    if len(ifmet_requests) == 0:
        return model
    
    model, _ = apply_algo(
        model,
        tokenizer,
        ifmet_requests,
        hparams_deeper,
        copy=False,
        return_orig_weights=False
    )
    
    return model

def edit_wise(
    model, 
    tokenizer, 
    items, 
    hparams, 
    loc_data, 
    loc_index, 
    apply_algo,
):
    """
    Caveat:
    1. the [EasyEdit](https://github.com/zjunlp/EasyEdit/blob/3488a66ee988d83ee7891a8abbbe6bcb24a77daf/easyeditor/editors/editor.py#L131-L136)
    repository is using `tokenizer.padding_side = "left"` when editing the model with WISE.
    2. the [EasyEdit](https://github.com/zjunlp/EasyEdit/blob/3488a66ee988d83ee7891a8abbbe6bcb24a77daf/easyeditor/editors/editor.py#L385-L410)
    repository is editing per request.
    """
    requests = []
    for edit_item in items:
        for i, request in enumerate(edit_item["requested_rewrite"]):
            request['prompt'] = request['prompt'].format(request['subject'])
            if isinstance(request["target_new"], dict):
                request['target_new'] = request['target_new']['str']
            request.update({
                'loc_prompt': loc_data[loc_index + i]['loc'] + ' ' + loc_data[loc_index + i]['loc_ans']
            })
            loc_index += 1
        requests.extend(edit_item["requested_rewrite"])
    hparams.batch_size = len(requests)
    tokenizer.padding_side = "left"
    for request in requests:
        _, _ = apply_algo(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=False
        )
    return loc_index
