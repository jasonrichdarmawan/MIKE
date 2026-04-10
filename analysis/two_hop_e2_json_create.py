import argparse
import ast
import json
from pathlib import Path

import pandas as pd

def build_multihop_question(r1_template, r2_template, e1_label, e3_type):
    prefix = "Who is " if e3_type == "human" else "What is "
    return prefix + r2_template.format(r1_template.format(e1_label)) + "?"


def build_singlehop_question(r_template, subject_label, answer_type):
    prefix = "Who is " if answer_type == "human" else "What is "
    return prefix + r_template.format(subject_label) + "?"

def convert_row_to_case(row):
    r1_template = row["r1_template"]
    r2_template = row["r2_template"]

    e1_label = str(row["e1_label"])
    e2_label = str(row["e2_label"])
    e3_label = str(row["e3_label"])
    
    e2_aliases = ast.literal_eval(row["e2_aliases"])
    e3_aliases = ast.literal_eval(row["e3_aliases"])
    
    # Unless e2_new / e3_new is provided, we keep the original label and aliases
    e2_label_new = e2_label
    e3_label_new = e3_label
    e2_aliases_new = e2_aliases
    e3_aliases_new = e3_aliases

    if  pd.notna(row.get("e2_label_new")):
        e2_label_new = row.get("e2_label_new")
        e2_aliases_new = ast.literal_eval(row["e2_aliases_new"])

    if  pd.notna(row.get("e3_label_new")):
        e3_label_new = row.get("e3_label_new")
        e3_aliases_new = ast.literal_eval(row["e3_aliases_new"])

    e2_label_new = str(e2_label_new)
    e3_label_new = str(e3_label_new)
    
    requested_rewrite = {
        "prompt": r1_template,
        "relation_id": str(row["r1"]),
        "target_new": {
            "str": e2_label_new,
            "id": str(row["e2_new"]),
        },
        "target_true": {
            "str": e2_label,
            "id": str(row["e2"]),
        },
        "subject": e1_label,
        "question": build_singlehop_question(str(row["r1_template"]), e1_label, str(row["e2_type"])),
    }
    
    new_single_hops = [
        {
            "question": build_singlehop_question(str(row["r1_template"]), e1_label, str(row["e2_type"])),
            "cloze": f"{r1_template.format(e1_label).capitalize()} is",
            "answer": e2_label_new,
            "answer_alias": e2_aliases_new,
        },
        {
            "question": build_singlehop_question(str(row["r2_template"]), e2_label_new, str(row["e3_type"])),
            "cloze": f"{r2_template.format(e2_label_new).capitalize()} is",
            "answer": e3_label_new,
            "answer_alias": e3_aliases_new,
        },
    ]
    
    cloze_q = r2_template.format(r1_template.format(e1_label))
    cloze_q = cloze_q[0].upper() + cloze_q[1:]

    return {
        "case_id": int(row["id"]),
        "row_role": str(row["row_role"]),
        "requested_rewrite": [requested_rewrite],
        "questions": [build_multihop_question(str(row["r1_template"]), str(row["r2_template"]), e1_label, str(row["e3_type"]))],
        "answer": e3_label,
        "answer_alias": e3_aliases,
        "new_answer": e3_label_new,
        "new_answer_alias": e3_aliases_new,
        "single_hops": [
            {
                "question": build_singlehop_question(str(row["r1_template"]), e1_label, str(row["e2_type"])),
                "cloze": f"{r1_template.format(e1_label).capitalize()} is",
                "answer": e2_label,
                "answer_alias": e2_aliases,
            },
            {
                "question": build_singlehop_question(str(row["r2_template"]), e2_label, str(row["e3_type"])),
                "cloze": f"{r2_template.format(e2_label).capitalize()} is",
                "answer": e3_label,
                "answer_alias": e3_aliases,
            },
        ],
        "new_single_hops": new_single_hops,
        "cloze_question": cloze_q + " is",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="datasets/two_hop_cf_e2.csv", type=str)
    parser.add_argument("--output_path", default="datasets/two_hop_cf_e2-cake.json", type=str)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    df = pd.read_csv(input_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    cases = [convert_row_to_case(row) for _, row in df.iterrows()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(cases)} cases to {output_path}")


if __name__ == "__main__":
    main()
