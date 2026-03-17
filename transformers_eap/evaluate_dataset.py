"""
This is a simple evaluation, the MCQ is not dynamic. So, make sure that no entity with citizenship "Australia" or "Japan" is included in the dataset. Otherwise, the MCQ evaluation will be flawed.
"""

# %%

from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import ast
from datasets import Dataset
from tqdm import tqdm
import torch

import re

# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%

BATCH_SIZE = 16

# MODEL_ID = "Qwen/Qwen3.5-4B"
# BASE_MODEL = False
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
BASE_MODEL = True

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# %%

# Single-Hop
SINGLE_HOP_CLOZE_TEMPLATES = {
    "P27": "{e1_label}'s country of citizenship is "
}
SINGLE_HOP_QUESTIONS_TEMPLATES = {
    "P27": "What is the country of citizenship of {e1_label}?"
}
SINGLE_HOP_MCQ_TEMPLATES = {
    "P27": {
        "template": "What is the country of citizenship of {e1_label}?\nA) Australia\nB) {e2_label}\nC) Japan",
        "answers": ["B", "B) {e2_label}", "{e2_label}"],
    }
}

# Attribute-Fact
ATTRIBUTE_FACT_CLOZE_TEMPLATES = {
    "P27": "Suppose Ellie Kemper lives in a houseboat, Keanu Reeves lives in an apartment, and {e1_label} lives in a cabin. Therefore, the person who lives in the cabin is a citizen of "
}
ATTRIBUTE_FACT_QUESTIONS_TEMPLATES = {
    "P27": "Suppose Ellie Kemper lives in a houseboat, Keanu Reeves lives in an apartment, and {e1_label} lives in a cabin. What is the country of citizenship of the person who lives in the cabin?"
}
ATTRIBUTE_FACT_MCQ_TEMPLATES = {
    "P27": {
        "template": "Suppose Ellie Kemper lives in a houseboat, Keanu Reeves lives in an apartment, and {e1_label} lives in a cabin. What is the country of citizenship of the person who lives in the cabin?\nA) Australia\nB) {e2_label}\nC) Japan",
        "answers": ["B", "B) {e2_label}", "{e2_label}"],
    }
}

# Fact-Attribute
FACT_ATTRIBUTE_CLOZE_TEMPLATES = {
    "P27": {
        "template": "In a book about different countries, Australia is mentioned on page 6, {e2_label} is mentioned on page 12, and Japan is mentioned on page 18. {e1_label}'s country of citizenship is listed on page ",
        "answers": ["12"],
    }
}
FACT_ATTRIBUTE_QUESTIONS_TEMPLATES = {
    "P27": {
        "template": "In a book about different countries, Australia is mentioned on page 6, {e2_label} is mentioned on page 12, and Japan is mentioned on page 18. On which page is {e1_label}'s country of citizenship listed?",
        "answers": ["12"],
    }
}
FACT_ATTRIBUTE_MCQ_TEMPLATES = {
    "P27": {
        "template": "In a book about different countries, Australia is mentioned on page 6, {e2_label} is mentioned on page 12, and Japan is mentioned on page 18. On which page is {e1_label}'s country of citizenship listed?\nA) 6\nB) 12\nC) 18",
        "answers": ["B", "B) 12", "12"],
    }
}

def _format_prompt(prompt, tokenizer, base_model=False):
    """
    Findings:
    - `tokenizer.apply_chat_template(..., enable_thinking=True)` needs more than max_new_tokens=1024. So, we disable thinking.
    """
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

def stage0_build_prompts(df: pd.DataFrame, base_model: bool = False):
    for col in ["e1_aliases", "e2_aliases", "e2_label"]:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    
    df["e2_answers"] = df.apply(lambda row: row["e2_label"] + row["e2_aliases"], axis=1)
    
    # Single-Hop Cloze
    df["e1_single_cloze_prompt"] = df.apply(lambda row: _format_prompt(SINGLE_HOP_CLOZE_TEMPLATES[row["r1"]].format(e1_label=row["e1_label"]), tokenizer, base_model=base_model), axis=1)
    
    # Single-Hop Question
    df["e1_single_question_prompt"] = df.apply(lambda row: _format_prompt(SINGLE_HOP_QUESTIONS_TEMPLATES[row["r1"]].format(e1_label=row["e1_label"]), tokenizer, base_model=base_model), axis=1)
    
    # Single-Hop MCQ
    df["e1_single_mcq_prompt"] = df.apply(lambda row: _format_prompt(SINGLE_HOP_MCQ_TEMPLATES[row["r1"]]["template"].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    df["e1_single_mcq_answers"] = df.apply(lambda row: [answer.format(e2_label=row["e2_label"][0]) for answer in SINGLE_HOP_MCQ_TEMPLATES[row["r1"]]["answers"]], axis=1)
    
    # Attribute-Fact Cloze
    df["e1_attribute_cloze_prompt"] = df.apply(lambda row: _format_prompt(ATTRIBUTE_FACT_CLOZE_TEMPLATES[row["r1"]].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    
    # Attribute-Fact Question
    df["e1_attribute_question_prompt"] = df.apply(lambda row: _format_prompt(ATTRIBUTE_FACT_QUESTIONS_TEMPLATES[row["r1"]].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    
    # Attribute-Fact MCQ
    df["e1_attribute_mcq_prompt"] = df.apply(lambda row: _format_prompt(ATTRIBUTE_FACT_MCQ_TEMPLATES[row["r1"]]["template"].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    df["e1_attribute_mcq_answers"] = df.apply(lambda row: [answer.format(e2_label=row["e2_label"][0]) for answer in ATTRIBUTE_FACT_MCQ_TEMPLATES[row["r1"]]["answers"]], axis=1)
    
    # Fact-Attribute Cloze
    df["e1_fact_cloze_prompt"] = df.apply(lambda row: _format_prompt(FACT_ATTRIBUTE_CLOZE_TEMPLATES[row["r1"]]["template"].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    df["e1_fact_cloze_answers"] = df.apply(lambda row: FACT_ATTRIBUTE_CLOZE_TEMPLATES[row["r1"]]["answers"], axis=1)
    
    # Fact-Attribute Question
    df["e1_fact_question_prompt"] = df.apply(lambda row: _format_prompt(FACT_ATTRIBUTE_QUESTIONS_TEMPLATES[row["r1"]]["template"].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    df["e1_fact_question_answers"] = df.apply(lambda row: FACT_ATTRIBUTE_QUESTIONS_TEMPLATES[row["r1"]]["answers"], axis=1)
    
    # Fact-Attribute MCQ
    df["e1_fact_mcq_prompt"] = df.apply(lambda row: _format_prompt(FACT_ATTRIBUTE_MCQ_TEMPLATES[row["r1"]]["template"].format(e1_label=row["e1_label"], e2_label=row["e2_label"][0]), tokenizer, base_model=base_model), axis=1)
    df["e1_fact_mcq_answers"] = df.apply(lambda row: [answer.format(e2_label=row["e2_label"][0]) for answer in FACT_ATTRIBUTE_MCQ_TEMPLATES[row["r1"]]["answers"]], axis=1)
    
    return df

def stage1_evaluate(rows, indices, model, tokenizer, prompt_col, answers_col, result_prefix, base_model: bool = False):
    """Generalized single-batch evaluator for stage1 checks.

    rows: dict-like batch from Dataset.map
    indices: list of indices for the current batch
    prompt_col: str column name containing prompts
    answers_col: str column name containing list(s) of acceptable answers
    result_key: str key to return in result dict (e.g. 'e1_single_correct')
    """
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(rows[prompt_col], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=1, do_sample=False,
            max_new_tokens=100,
            stop_strings=["\n\n", "Question:"] if base_model else None,
            tokenizer=tokenizer if base_model else None,
        )

    predictions = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    def _extract_answer_text(prediction, base_model):
        if base_model:
            m = re.search(r"(.*?)</answer>", prediction, re.IGNORECASE | re.DOTALL)
            text = m.group(1) if m else prediction
        else:
            m = re.search(r"<answer>(.*?)</answer>", prediction, re.IGNORECASE | re.DOTALL)
            text = m.group(1) if m else prediction
        return text.strip()

    correct = []
    for prediction, answers in zip(predictions, rows[answers_col]):
        text = _extract_answer_text(prediction, base_model).lower()
        matched = False
        for ans in answers:
            a = ans.strip().lower()
            if not a:
                continue
            if a in text or text.startswith(a) or text == a:
                matched = True
                break
        correct.append(matched)
    
    tqdm.write(f"Index: {indices[0]}")
    tqdm.write(f"{'=' * 100}\nPrompt: \n{rows[prompt_col][0]}\n{'-' * 100}")
    tqdm.write(f"{'-' * 100}\nPrediction: \n{predictions[0]}\n{'=' * 100}")
    tqdm.write(f"Acceptable Answers: {rows[answers_col][0]}")
    tqdm.write(f"Correct: {correct[0]}")
    
    return {f"{result_prefix}_correct": correct, f"{result_prefix}_prediction": predictions}

df = pd.read_csv("entity_citizenship.csv")
df = stage0_build_prompts(df, base_model=BASE_MODEL)
ds = Dataset.from_pandas(df)
for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating"):
    end = min(i + BATCH_SIZE, len(ds))
    indices = range(i, end)
    batch = ds.select(indices)
    
    # Single-Hop Cloze
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_single_cloze_prompt", "e2_answers", "e1_single_cloze", BASE_MODEL)
    df.loc[indices, ["e1_single_cloze_correct", "e1_single_cloze_prediction"]] = list(zip(result["e1_single_cloze_correct"], result["e1_single_cloze_prediction"]))
    
    # Single-Hop Question
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_single_question_prompt", "e2_answers", "e1_single_question", BASE_MODEL)
    df.loc[indices, ["e1_single_question_correct", "e1_single_question_prediction"]] = list(zip(result["e1_single_question_correct"], result["e1_single_question_prediction"]))
    
    # Single-Hop MCQ
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_single_mcq_prompt", "e1_single_mcq_answers", "e1_single_mcq", BASE_MODEL)
    df.loc[indices, ["e1_single_mcq_correct", "e1_single_mcq_prediction"]] = list(zip(result["e1_single_mcq_correct"], result["e1_single_mcq_prediction"]))
    
    # Attribute-Fact Cloze
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_attribute_cloze_prompt", "e2_answers", "e1_attribute_cloze", BASE_MODEL)
    df.loc[indices, ["e1_attribute_cloze_correct", "e1_attribute_cloze_prediction"]] = list(zip(result["e1_attribute_cloze_correct"], result["e1_attribute_cloze_prediction"]))
    
    # Attribute-Fact Question
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_attribute_question_prompt", "e2_answers", "e1_attribute_question", BASE_MODEL)
    df.loc[indices, ["e1_attribute_question_correct", "e1_attribute_question_prediction"]] = list(zip(result["e1_attribute_question_correct"], result["e1_attribute_question_prediction"]))
    
    # Attribute-Fact MCQ
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_attribute_mcq_prompt", "e1_attribute_mcq_answers", "e1_attribute_mcq", BASE_MODEL)
    df.loc[indices, ["e1_attribute_mcq_correct", "e1_attribute_mcq_prediction"]] = list(zip(result["e1_attribute_mcq_correct"], result["e1_attribute_mcq_prediction"]))
    
    # Fact-Attribute Cloze
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_fact_cloze_prompt", "e1_fact_cloze_answers", "e1_fact_cloze", BASE_MODEL)
    df.loc[indices, ["e1_fact_cloze_correct", "e1_fact_cloze_prediction"]] = list(zip(result["e1_fact_cloze_correct"], result["e1_fact_cloze_prediction"]))
    
    # Fact-Attribute Question
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_fact_question_prompt", "e1_fact_question_answers", "e1_fact_question", BASE_MODEL)
    df.loc[indices, ["e1_fact_question_correct", "e1_fact_question_prediction"]] = list(zip(result["e1_fact_question_correct"], result["e1_fact_question_prediction"]))
    
    # Fact-Attribute MCQ
    result = stage1_evaluate(batch, indices, model, tokenizer, "e1_fact_mcq_prompt", "e1_fact_mcq_answers", "e1_fact_mcq", BASE_MODEL)
    df.loc[indices, ["e1_fact_mcq_correct", "e1_fact_mcq_prediction"]] = list(zip(result["e1_fact_mcq_correct"], result["e1_fact_mcq_prediction"]))

print(f"Single-Hop Cloze Accuracy: {df['e1_single_cloze_correct'].mean() * 100:.2f}%")
print(f"Single-Hop Question Accuracy: {df['e1_single_question_correct'].mean() * 100:.2f}%")
print(f"Single-Hop MCQ Accuracy: {df['e1_single_mcq_correct'].mean() * 100:.2f}%")
single_hop_overall_accuracy = (df['e1_single_cloze_correct'] & df['e1_single_question_correct'] & df['e1_single_mcq_correct']).mean()
print(f"Single-Hop Overall Accuracy: {single_hop_overall_accuracy* 100:.2f}%")

print(f"Attribute-Fact Cloze Accuracy: {df['e1_attribute_cloze_correct'].mean() * 100:.2f}%")
print(f"Attribute-Fact Question Accuracy: {df['e1_attribute_question_correct'].mean() * 100:.2f}%")
print(f"Attribute-Fact MCQ Accuracy: {df['e1_attribute_mcq_correct'].mean() * 100:.2f}%")
attribute_fact_overall_accuracy = (df['e1_attribute_cloze_correct'] & df['e1_attribute_question_correct'] & df['e1_attribute_mcq_correct']).mean()
print(f"Attribute-Fact Overall Accuracy: {attribute_fact_overall_accuracy* 100:.2f}%")

print(f"Fact-Attribute Cloze Accuracy: {df['e1_fact_cloze_correct'].mean() * 100:.2f}%")
print(f"Fact-Attribute Question Accuracy: {df['e1_fact_question_correct'].mean() * 100:.2f}%")
print(f"Fact-Attribute MCQ Accuracy: {df['e1_fact_mcq_correct'].mean() * 100:.2f}%")
fact_attribute_overall_accuracy = (df['e1_fact_cloze_correct'] & df['e1_fact_question_correct'] & df['e1_fact_mcq_correct']).mean()
print(f"Fact-Attribute Overall Accuracy: {fact_attribute_overall_accuracy* 100:.2f}%")

cloze_overall_accuracy = (df['e1_single_cloze_correct'] & df['e1_attribute_cloze_correct'] & df['e1_fact_cloze_correct']).mean()
print(f"Cloze Overall Accuracy: {cloze_overall_accuracy* 100:.2f}%")
mcq_overall_accuracy = (df['e1_single_mcq_correct'] & df['e1_attribute_mcq_correct'] & df['e1_fact_mcq_correct']).mean()
print(f"MCQ Overall Accuracy: {mcq_overall_accuracy* 100:.2f}%")
question_overall_accuracy = (df['e1_single_question_correct'] & df['e1_attribute_question_correct'] & df['e1_fact_question_correct']).mean()
print(f"Question Overall Accuracy: {question_overall_accuracy* 100:.2f}%")
overall_accuracy = (
    df['e1_single_cloze_correct'] & df['e1_single_question_correct'] & df['e1_single_mcq_correct'] & 
    df['e1_attribute_cloze_correct'] & df['e1_attribute_question_correct'] & df['e1_attribute_mcq_correct'] & 
    df['e1_fact_cloze_correct'] & df['e1_fact_question_correct'] & df['e1_fact_mcq_correct']).mean()
print(f"Overall Accuracy: {overall_accuracy* 100:.2f}%")

# %%
