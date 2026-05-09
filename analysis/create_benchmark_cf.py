# %%

import sys
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# %%

from argparse import ArgumentParser
import pandas as pd
import ast

from IPython.display import display

# %%

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path")
    args = parser.parse_args()
    
    args.swapping_strategy = args.input_path.split("/")[-1].split(".")[0].split("_")[-1]
    if args.swapping_strategy not in ["e2", "e3"]:
        raise ValueError(f"Invalid swapping strategy: {args.swapping_strategy}")
    
    args.output_path = args.input_path.replace("two_hop_cf", "two_hop_cf_benchmark")
    
    return args

def create_subset_cf(df: pd.DataFrame, swapping_strategy: str):
    """
    Note: Only evaluate the rows with "row_role" column with value "propagated_eval" after the model learn the counterfactual of the rows with "row_role" column with value "edit". This is because the rows with "row_role" column with value "propagated_eval" are designed to evaluate the model's ability to generalize the knowledge learned from the counterfactual examples, and they are not necessarily valid for evaluation before the model learns the counterfactual examples.
    
    - a row with "row_role" column with value "edit":
      - if swapping_strategy is "e2":
        - The row is valid for editing if the pre-edited model can answer the second hop of the "swapped_with" row correctly.
      - elif swapping_strategy is "e3":
        - The row is valid for editing if the pre-edited model can answer the first hop correctly.
    - a row with "row_role" column with value "propagated_eval":
      - The row is valid for evaluation if the propagation source is available in the dataset.
      - if "propagated_from" column with 2 propagation steps, it is a valid row for evaluation regardless whether the pre-edited model can answer the first hop, the second hop, or the composition correctly. 
      - if "propagated_from" column with 1 propagation step, it is a valid row for evaluation only if the pre-edited model can answer the first hop correctly.
    """
    stats = {}
    
    df = df.copy()
    
    orig_index = list(df.index)
    
    # 1) drop edit rows that are not valid for editing
    edit_ids = list(df[df['row_role'] == "edit"].index)
    
    keep_ids: list[int] = []
    
    for edit_id in edit_ids:
        row_i = df.loc[edit_id]
        
        if pd.isna(row_i["swapped_with"]):
            logging.debug(f"Warning: edit row {edit_id} has no swapped_with")
            continue
        
        # Note: If this is not removed, it will bias the analysis.
        # For example,
        # we edit [e1, r1] to output [e2_new], which will affect [e1, r1, e2_new, r2, e3_new].
        # In this case, if e3_new == e3:
        # [r2, e1] is an entity shortcut for e3, then the analysis will be biased if we keep this edit row.
        # [r2, r1] is a relation shortcut for e3, then the analysis will be biased if we keep this edit row.
        if pd.isna(row_i["e3_new"]) and (row_i["entity_shortcut_correct"] == True or row_i["relation_shortcut_correct"] == True):
            continue
        
        row_j = df.loc[row_i["swapped_with"]]
        
        # Note: If this is not removed, it will bias the analysis.
        # For example,
        # we edit [e1, r1] to output [e2_new], which will affect [e1, r1, e2_new, r2, e3_new].
        # In this case, if e3_new != e3:
        # [r2, e1] is an entity shortcut for e3_new, then the analysis will be biased if we keep this.
        # [r2, r1] is a relation shortcut for e3_new, then the analysis will be biased if we keep this edit row.
        if pd.notna(row_i["e3_new"]) and (row_j["entity_shortcut_correct"] == True or row_j["relation_shortcut_correct"] == True):
            continue
        
        if swapping_strategy == "e2":
            # If the swapped_with row can't answer the second hop, then the edit row is not valid for editing.
            if row_j["second_hop_correct"] == False:
                continue
        
        elif swapping_strategy == "e3":
            # If the row can't answer the first hop, then the edit row is not valid for editing.
            if row_i["first_hop_correct"] == False:
                continue
        
        keep_ids.append(edit_id)
    
    df = df.loc[keep_ids].copy()

    # 2) drop propagated_eval rows that are not valid for evaluation
    edit_ids = list(df[df['row_role'] == "edit"].index)
    propagated_ids = list(df[df['row_role'] == "propagated_eval"].index)
    
    dep_map = {p: set([t[1] for t in df.at[p, "propagated_from"]]) for p in propagated_ids}
    
    placed = set()
    out_order: list[int] = []
    
    remaining_propagated_ids = set(propagated_ids)
    
    for edit_id in edit_ids:
        out_order.append(edit_id)
        placed.add(edit_id)
        
        for propagated_id in sorted(list(remaining_propagated_ids)):
            propagate_source_ids = dep_map[propagated_id]
            
            if not propagate_source_ids.issubset(placed):
                continue
            
            if len(propagate_source_ids) == 1:
                if df.at[propagated_id, "first_hop_correct"] == False:
                    continue
            
            out_order.append(propagated_id)
            placed.add(propagated_id)
            remaining_propagated_ids.remove(propagated_id)
    
    removed_ids = set(orig_index) - set(out_order)
    
    df = df.loc[out_order].copy()
    
    stats["original_length"] = len(orig_index)
    stats["removed_length"] = len(removed_ids)
    stats["remaining_length"] = len(out_order)
    stats["edit_rows"] = len(df[df['row_role'] == "edit"])
    stats["propagated_eval_rows"] = len(df[df['row_role'] == "propagated_eval"])
    
    return stats, df

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.input_path)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.set_index("id")
    df["propagated_from"] = df.apply(lambda row: ast.literal_eval(row["propagated_from"]) if row["row_role"] == "propagated_eval" else row["propagated_from"], axis=1)
    df["swapped_with"] = df["swapped_with"].astype("Int64")
    stats, df = create_subset_cf(df, args.swapping_strategy)
    
    output = " | ".join([f"{k}: {v}" for k, v in stats.items()])
    print(f"Create subset counterfactual complete | {output}")
    
    columns = ["e1_label", "r1_type", "e2_label", "e2_label_new", "r2_type", "e3_label", "e3_label_new", "row_role", "swapped_with", "propagated_from"]
    display(df[columns])
    
    df.to_csv(args.output_path)
    print(f"Saved subset counterfactual to {args.output_path}")

# %%