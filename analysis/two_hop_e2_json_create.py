# %%

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

# %%

def build_multihop_question(r1_template, r2_template, e1_label, e3_type):
    """
    r1_template: "the work that features {}"
    r2_template: "the country of origin of {}"
    e1_label: "George Washington"
    e3_type: "human" or "non-human"
    """
    prefix = ""
    if e3_type == "human":
        prefix = "Who is"
    else:
        prefix = "What is"
    return f"{prefix} {r2_template.format(r1_template.format(e1_label))}?"


def build_singlehop_question(r_template, subject_label, answer_type):
    """
    r_template: "the work that features {}"
    subject_label: "George Washington"
    answer_type: "video game"
    """
    prefix = ""
    if answer_type == "human":
        prefix = "Who is"
    else:
        prefix = "What is"
    return f"{prefix} {r_template.format(subject_label)}?"

def convert_row_to_case(row):
    r1_template = row["r1_template"]
    r2_template = row["r2_template"]

    e1_label = row["e1_label"]
    e2_label = row["e2_label"]
    e3_label = row["e3_label"]
    
    e2_aliases = ast.literal_eval(row["e2_aliases"])
    e3_aliases = ast.literal_eval(row["e3_aliases"])
    
    # Unless e2_new / e3_new is provided (or is NaN), keep the original label and aliases
    def _get_new(key, default, literal_eval=False):
        val = row[key]
        if pd.isna(val):
            return default
        if not literal_eval:
            return val
        return ast.literal_eval(val)

    e2_new = _get_new("e2_new", row["e2"])
    e2_label_new = _get_new(key="e2_label_new", default=e2_label)
    e2_aliases_new = _get_new(key="e2_aliases_new", default=e2_aliases, literal_eval=True)

    e3_label_new = _get_new(key="e3_label_new", default=e3_label)
    e3_aliases_new = _get_new(key="e3_aliases_new", default=e3_aliases, literal_eval=True)

    requests = []
    if row["row_role"] == "edit":
        request = {
            "swapped_with": row["swapped_with"],
            "prompt": f"{r1_template} is",
            "relation_id": row["r1"],
            "target_new": {
                "str": e2_label_new,
                "id": e2_new,
            },
            "target_true": {
                "str": e2_label,
                "id": row["e2"],
            },
            "subject": e1_label,
            "question": build_singlehop_question(
                r_template=row["r1_template"],
                subject_label=e1_label, 
                answer_type=row["e2_type"],
            ),
        }
        requests.append(request)
    
    single_hops = [
        {
            "question": build_singlehop_question(
                r_template=row["r1_template"],
                subject_label=e1_label,
                answer_type=row["e2_type"]
            ),
            "cloze": f"{r1_template.format(e1_label)} is",
            "answer": e2_label,
            "answer_alias": e2_aliases,
        },
        {
            "question": build_singlehop_question(
                r_template=row["r2_template"],
                subject_label=e2_label,
                answer_type=row["e3_type"]
            ),
            "cloze": f"{r2_template.format(e2_label)} is",
            "answer": e3_label,
            "answer_alias": e3_aliases,
        },
    ]
    
    new_single_hops = [
        {
            "question": build_singlehop_question(
                r_template=row["r1_template"],
                subject_label=e1_label,
                answer_type=row["e2_type"],
            ),
            "cloze": f"{r1_template.format(e1_label)} is",
            "answer": e2_label_new,
            "answer_alias": e2_aliases_new,
        },
        {
            "question": build_singlehop_question(
                r_template=row["r2_template"],
                subject_label=e2_label_new,
                answer_type=row["e3_type"]
            ),
            "cloze": f"{r2_template.format(e2_label_new)} is",
            "answer": e3_label_new,
            "answer_alias": e3_aliases_new,
        },
    ]
    propagated_from = row["propagated_from"]
    if not (isinstance(propagated_from, float) and pd.isna(propagated_from)) \
        and propagated_from is not None:
        for (swap_type, swapped_with) in propagated_from:
            if swap_type == "e2_new":
                new_single_hops[0].update({
                    "propagated_from": {
                        "swap_type": swap_type,
                        "swapped_with": swapped_with,
                    }
                })
            elif swap_type == "e3_new":
                new_single_hops[1].update({
                    "propagated_from": {
                        "swap_type": swap_type,
                        "swapped_with": swapped_with,
                    }
                })
            else:
                raise ValueError(f"Unknown swap_type: {swap_type}")

    return {
        "case_id": row["id"],
        "row_role": row["row_role"],
        "requested_rewrite": requests,
        "questions": [
            build_multihop_question(
                r1_template=row["r1_template"],
                r2_template=row["r2_template"],
                e1_label=e1_label,
                e3_type=row["e3_type"]
            )
        ],
        "answer": e3_label,
        "answer_alias": e3_aliases,
        "new_answer": e3_label_new,
        "new_answer_alias": e3_aliases_new,
        "single_hops": single_hops,
        "new_single_hops": new_single_hops,
        "cloze_question": f"{r2_template.format(r1_template.format(e1_label))} is",
    }

# if __name__ == "__main__":
#     import sys
#     sys.argv = [
#         "two_hop_e2_json_create.py",
#         "--input_path", "../datasets/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_benchmark_e2.csv",
#         "--output_path", "../datasets/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_benchmark_e2-clke.json",
#     ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    df = pd.read_csv(input_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    df["propagated_from"] = df.apply(lambda row:
        ast.literal_eval(row["propagated_from"])
        if row["row_role"] == "propagated_eval"
        else row["propagated_from"], axis=1
    )
    df["swapped_with"] = df["swapped_with"].astype("Int64")

    cases = [convert_row_to_case(row) for _, row in df.iterrows()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(cases)} cases to {output_path}")

# %%
