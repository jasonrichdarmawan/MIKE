import ast

import pandas as pd
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, \
    LlamaForCausalLM, GemmaForCausalLM, Qwen2ForCausalLM

from jaxtyping import Float

import os

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
    elif type(model) is LlamaForCausalLM:
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    elif type(model) is GemmaForCausalLM:
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    elif type(model) is Qwen2ForCausalLM:
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_attention_layers_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}.attn" for i in range(model.config.num_hidden_layers)]
    elif type(model) is LlamaForCausalLM:
        return [f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_mlp_layers_names(model):
    if type(model) is GPT2LMHeadModel:
        return [f"transformer.h.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    elif type(model) is LlamaForCausalLM or type(model) is Qwen2ForCausalLM:
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
    elif type(model) is LlamaForCausalLM:
        return [model.model.layers[l].self_attn for l in range(bot, top)]
    elif type(model) is GemmaForCausalLM:
        return [model.model.layers[l].self_attn for l in range(bot, top)]
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def get_norm_module(model):
    if type(model) is GPT2LMHeadModel:
        return model.transformer.ln_f
    elif type(model) is LlamaForCausalLM:
        return model.model.norm
    elif type(model) is GemmaForCausalLM:
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
    elif type(model) is Qwen2ForCausalLM:
        return True
    elif type(model) is GemmaForCausalLM:
        return True
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def decode_generated(tokenizer, generated: Float[Tensor, "batch_size seq_len"], prompts: list[str]) -> list[str]:
    text = tokenizer.batch_decode(generated, skip_special_tokens=True) # list of str, len=batch_size
    pred = [t[len(p):].strip().replace("\n", " ") for t, p in zip(text, prompts)]
    return pred


def get_answers(entry, key: str):
    aliases = ast.literal_eval(entry[f"{key}_aliases"])
    aliases = [a for a in aliases if len(a) > 1]
    entity = entry[f"{key}_label"]
    return {
        f"{key}_answers": [entity] + aliases,
    }


def check_answer_in_pred(pred: str, answers: list[str]):
    pred = pred.lower()
    return any([a.lower() in pred for a in answers])


def generate_and_test_answers(model, tokenizer, task: str, prompts: list[str], ids: list[int], answers: list[list[str]]):
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
                                   pad_token_id=tokenizer.eos_token_id, max_new_tokens=10) # shape (batch_size, seq_len)
    predictions = decode_generated(tokenizer, generated, prompts) # list of str, len=batch_size
    prompt = tokenizer.decode([t for t in inputs.input_ids[0] if t != tokenizer.pad_token_id], skip_special_tokens=False)
    print(f"Task: {task}\nid: {ids[0]}\nPrompt: {prompt}\nPrediction: {predictions[0]}\nAnswers: {answers[0]}\nResult: {check_answer_in_pred(predictions[0], answers[0])}")
    print("-----------------------------------")
    return [check_answer_in_pred(p, a) for p, a in zip(predictions, answers)]


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
