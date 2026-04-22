import ast

import pandas as pd
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, \
    LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM

from jaxtyping import Float

import os

import re

HF_TOKEN = os.getenv("HF_TOKEN")

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, device="cuda"):
    if "70b" in model_name or "70B" in model_name:
        return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto",
                                                    token=HF_TOKEN)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN).to(device)


def get_layer_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}" for i in range(model.config.num_hidden_layers)]
    elif type(model) in [LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM]:
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_attention_layers_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}.attn" for i in range(model.config.num_hidden_layers)]
    elif type(model) in [LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM]:
        return [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_mlp_layers_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    elif type(model) in [LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM]:
        return [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_attention_modules(model, layer: int, k=0):
    """
    Caveat: returns attention modules from layers [layer - k, layer + k]
    """
    bot = max(0, layer - k)
    top = min(layer + k + 1, model.config.num_hidden_layers)
    if type(model) is GPT2LMHeadModel:
        return [model.transformer.h[l].attn for l in range(bot, top)]
    elif type(model) in [LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM]:
        return [model.model.layers[l].self_attn for l in range(bot, top)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_norm_module(model):
    if type(model) is GPT2LMHeadModel:
        return model.transformer.ln_f
    elif type(model) in [LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM]:
        return model.model.norm
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_prepend_space(model):
    if type(model) is GPT2LMHeadModel:
        return True
    elif type(model) is LlamaForCausalLM:
        if "Llama-3" in model.config._name_or_path:
            return True
        elif "Llama-2" in model.config._name_or_path:
            return False
    elif type(model) in [Qwen2ForCausalLM, Qwen3ForCausalLM, Cohere2ForCausalLM]:
        return True
    elif type(model) is GemmaForCausalLM:
        return True
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def decode_generated(tokenizer, input_ids: Float[Tensor, "batch_size seq_len"], generated_ids: Float[Tensor, "batch_size seq_len"]) -> list[str]:
    """
    Caveat:
    - Biran et al. [1] use batch decoding, remove the prompt from the generated text, and then replace ("\n", " ") to get the final answer. However, we can slice the prompt's input ids from the generated ids to get the answer, which is more robust to chat template (e.g. using the <think></think> tag which is a special tokens). We also keep the "\n" in the answer since some answers may contain newlines (e.g., lists).
    
    [1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
    """
    predictions = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True) # list of str, len=batch_size
    return predictions


def get_answers(entry, key: str):
    aliases = ast.literal_eval(entry[f"{key}_aliases"])
    aliases = [a for a in aliases if len(a) > 1]
    entity = entry[f"{key}_label"]
    return {
        f"{key}_answers": [entity] + aliases,
    }

def _format_prompt(prompt: str, tokenizer, base_model = False):
    if base_model:
        few_shot_prompt = (
            "Question: What is the country of citizenship of Sir Henry Wilson, 1st Baronet?\n"
            "Answer: <answer>United Kingdom</answer>\n\n"
            f"Question: {prompt}\n"
            "Answer: <answer>"
        )
        return few_shot_prompt
    
    system_content = ""
    system_content += "1. Only respond with the answer to the question. Do not include any explanations or additional text."
    system_content += "\n2. Wrap your final answer in XML tags like this: <answer>ANSWER_HERE</answer>."""
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def _extract_answer_text_xml_tag(prediction: str, base_model: bool):
    """
    Caveat:
    - If the model does not generate the <answer> tag, we return the whole prediction as the answer. This may happen if the model.generate(max_new_tokens) is too small. However, this is not ideal. For example, if the model generate "It cannot be answered...", but the list of correct answers is ["Canada", "CA"], then we will incorrectly mark the answer as correct.
    """
    if base_model:
        m = re.search(r"(.*?)</answer>", prediction, re.IGNORECASE | re.DOTALL)
        text = m.group(1) if m else prediction
    else:
        m = re.search(r"<answer>(.*?)</answer>", prediction, re.IGNORECASE | re.DOTALL)
        text = m.group(1) if m else prediction
    return text.strip()

def check_answer_in_pred(pred: str, answers: list[str]):
    pred = pred.lower()
    return any(a.lower() in pred for a in answers)

def generate_and_test_answers(model, tokenizer, task: str, prompts: list[str], ids: list[int], answers: list[list[str]], check_answer_fn: callable, base_model=False, base_model_stop_strings=["\n\n", "Question:"]):
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
            pad_token_id=tokenizer.eos_token_id, max_new_tokens=200,
            stop_strings=base_model_stop_strings if base_model else None,
            tokenizer=tokenizer
        ) # shape (batch_size, seq_len)
    predictions = decode_generated(tokenizer, inputs.input_ids, generated) # list of str, len=batch_size
    prompt = tokenizer.decode([t for t in inputs.input_ids[0] if t != tokenizer.pad_token_id], skip_special_tokens=False)
    print(
        f"{'=' * 100}\n"
        f"Task: {task}\n"
        f"id: {ids[0]}\n"
        f"{'-' * 100}\nPrompt:\n{prompt}\n{'-' * 100}\n"
        f"{'-' * 100}\nPrediction:\n{predictions[0]}\n{'-' * 100}\n"
        f"{'-' * 100}\nAnswer:\n{answers[0]}\n{'-' * 100}\n"
        f"Result: {check_answer_fn(predictions[0], answers[0])}\n"
        f"{'=' * 100}\n"
    )
    return [check_answer_fn(p, a) for p, a in zip(predictions, answers)]


def print_dataset_statistics(dataset):
    print("Dataset statistics:")
    for target in ["e1", "r1", "e2", "r2", "e3"]:
        proportions = dataset[f"{target}_type"].value_counts(normalize=True) * 100
        counts = dataset[f"{target}_type"].value_counts()
        stats = pd.merge(counts, proportions, left_index=True, right_index=True)
        print(stats)


def rebalance_dataset(df, key="e2_type", size=100, secondary_key=None):
    if secondary_key is None:
        balanced = df.groupby(key).apply(lambda x: x.sample(min(size, len(x)))).reset_index(drop=True)
    else:
        balanced = df.groupby([key, secondary_key]).apply(lambda x: x.sample(min(size // 2, len(x)))).reset_index(drop=True)
    balanced = balanced.sort_values("id")
    return balanced


def last_relation_word(relation: str):
    return relation.split(" ")[-3]
