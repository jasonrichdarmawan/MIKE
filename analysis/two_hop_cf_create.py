"""
---
description: Clear pseudo algorithm for generating two-hop counterfactuals with conflict checks.

- Dataset info:
  - (e1, r1) have multiple e2 candidates across rows
    ```python
    mask = df.groupby(["e1", "r1"])["e2"].transform("nunique") > 1
    display(df[mask])
    ```
  - (e1, r1, e2, r2) have multiple e3 candidates across rows:
    - (Google, founded by, Sergey Brin, spouse, Anne Wojcicki)
    - (Google, founded by, Sergey Brin, spouse, Nicole Shanahan)
    ```python
    mask = df.groupby(["e1", "r1", "e2", "r2"])["e3"].transform("nunique") > 1
    display(df[mask])
    ```
  - (e1, r1, ?, r2, ?) have multiple (r1, e2) and (r2, e3) candidates across rows:
    - (ABBA, founded by, Björn Ulvaeus, spouse, Agnetha Fältskog)
    - (ABBA, founded by, Benny Andersson, spouse, Anni-Frid Lyngstad)
    ```python
    mask = (df.groupby(["e1", "r1", "r2"])["e2"].transform("nunique") > 1) & (df.groupby(["e1", "r1", "r2"])["e3"].transform("nunique") > 1)
    df[mask][["e1_label", "r1_type", "e2_label", "r2_type", "e3_label"]]
    ```
  - (e2, r2, e3) are in (e1, r1, e2) of other rows
    - (Stephen Harper, country of citizenship, Canada, head of state, Charles III of the United Kingdom)
    - (Canada, head of state, Charles III of the United Kingdom, father, 'Prince Philip, Duke of Edinburgh')
    ```python
    valid_knowledge = set(zip(df["e2"], df["r2"], df["e3"]))
    mask = df.apply(lambda row: (row["e1"], row["r1"], row["e2"]) in valid_knowledge, axis=1)
    display(df[mask])
    ```
  - the same (r1, e2, r2, e3) have different e1 across rows:
    - (Google, founded by, Sergey Brin, spouse, Anne Wojcicki)
    - (Google, founded by, Sergey Brin, spouse, Nicole Shanahan)
    - (Alphabet Inc., founded by, Sergey Brin, spouse, Anne Wojcicki)
    - (Alphabet Inc., founded by, Sergey Brin, spouse, Nicole Shanahan)
    ```python
    mask = df.groupby(["r1", "e2", "r2", "e3"])["e1"].transform("nunique") > 1
    display(df[mask][["e1_label", "r1_type", "e2_label", "r2_type", "e3_label"]])
    ```
  - The consequence of (e1, r1) and (e2, r2) having multiple candidates is that the (e1, r1, e2, r2) question have ambiguous answer, which makes the row invalid for evaluation.
    - (e1, r1, e2, r2, e3) have incorrect facts in the original dataset, e.g. `df.loc[63226]`:
      - "r1_template" is "the work that features WALL·E is" (e2 = "Disney Dreamlight Valley", which is correct)
      - "r2_tempalte" is "the country of origin of Disney Dreamlight Valley is" (e3 = "Canada", which is correct)
      - "source_prompt" is "the country of origin of the work that features WALL·E is" (e3 = "Canada" instead of "United States")
  - In 2026, some facts are stale:
    - (Sergey Brin, spouse, Anne Wojcicki)
    - (Sergey Brin, spouse, Nicole Shanahan)
  - Conclusion:
    - when swapping_strategy == "e2", we removed (e2, r2) that have multiple e3 candidates to avoid having invalid row because the second hop is ambiguous.
      - (Sergey Brin, spouse, Anne Wojcicki)
      - (Sergey Brin, spouse, Nicole Shanahan)
      ```python
      mask = df.groupby(["e2", "r2"])["e3"].transform("nunique") == 1
      display(df[mask])
      ```
    - when swapping_strategy == "e3", we removed (e1, r1) that have multiple e2 candidates to avoid having invalid row because the first hop fact is ambiguous
      - (Jenna Jameson, spouse, Jay Grdina)
      - (Jenna Jameson, spouse, Brad Armstrong)
      ```python
      mask = df.groupby(["e1", "r1"])["e2"].transform("nunique") > 1
      display(df[mask])
      ```
    - we removed (e1, r1) and (e2, r2) that have multiple e2 or e3 (evaluated globally), respectively to avoid propagating rows with ambiguous facts
    - we only process relations with rows that are unlikely to have stale facts, e.g. "mother", "father", "present in work", "notable work", "location of formation", "founded by", "country of origin"
    ```python
    relations = ["mother", "father", "present in work", "notable work", "location of formation", "founded by", "country of origin"]
    mask = (df.groupby(["e1", "r1"])["e2"].transform("nunique") == 1) & (df.groupby(["e2", "r2"])["e3"].transform("nunique") == 1) & (df["r1_type"].isin(relations)) & (df["r2_type"].isin(relations))
    df_2 = df[mask]
    ```
- swapping_strategy == "e2":
  - Original:
    - (Louis IX of France, mother, Blance of Castile, father, Alfonso VIII of Castile)
    - (Barack Obama, mother, Stanley Ann Dunham, father, Stanley Armour Dunham)
    ```python
    mask = (df_2["r1_type"] == "mother") & (df_2["r2_type"] == "father")
    df_2[mask][["e1_label", "r1_type", "e2_label", "r2_type", "e3_label"]]
    ```
  - Swapped:
    - (Louis IX of France, mother, Stanley Ann Dunham, father, Stanley Armour Dunham)
    - (Barack Obama, mother, Blance of Castile, father, Alfonso VIII of Castile)
  - To avoid causing the edited row become invalid because a later row edit the second hop (e2, r2). These facts must not be swapped:
    - (Stanley Ann Dunham, father, Stanley Armour Dunham)
    - (Blance of Castile, father, Alfonso VIII of Castile)
  - Conclusion:
    - when swapping e2:
      - the [e2_j, r2_i, e2_i] and [e2_i, r2_j, e2_j] must be protected to avoid causing the edited row become invalid because a later row edit the second hop (e2, r2)
      - the [(e1, r1) and (e2_new, r2)] or (e2, r2) of other rows must be updated with current facts to avoid evaluating rows with stale (entity, relation).
    - when evaluating knowledge editing methods performance:
      - before editing, the model must be able to answer row with row_role "edit":
        - (e2_i, r2_i, e3_i)
        - (e2_j, r2_j, e3_j)
      - the model must learn:
        - (e1_i, r1_i, e2_j)
        - (e1_j, r1_j, e2_i)
      - when evaluating the row with row_role "propagated_eval":
        - the rows with row_role "edit" must be learnt first
        - if "propagated_from" is [("e2_new", a), ("e3_new", b)]
        - if "propagated_from" is [("e3_new", a)], the model must be able to answer (e1, r1, e2) of the row
        - if not, then it's an invalid row
    - swapping_strategy == "e2" does 1 edit.
    - We can analyze whether the model think about e2 and output e3 when asked (e1, r1, r2):
      - (Louis IX of France, mother, Stanley Ann Dunham, father, Stanley Armour Dunham)
        - The father of the mother of Louis IX of France is
- swapping_strategy == "e3":
  - Original
    - (Louis IX of France, mother, Blance of Castile, father, Alfonso VIII of Castile)
    - (Barack Obama, mother, Stanley Ann Dunham, father, Stanley Armour Dunham)
    ```python
    mask = (df_2["r1_type"] == "mother") & (df_2["r2_type"] == "father")
    df_2[mask][["e1_label", "r1_type", "e2_label", "r2_type", "e3_label"]]
    ```
  - Swapped:
    - (Louis IX of France, mother, Blance of Castile, father, Stanley Armour Dunham)
    - (Barack Obama, mother, Stanley Ann Dunham, father, Alfonso VIII of Castile)
  - To avoid causing the edited row become invalid because a later row edit the first hop (entity, relation). These facts must not be swapped:
    - (Louis IX of France, mother, Blance of Castile)
    - (Barack Obama, mother, Stanley Ann Dunham)
  - Conclusion:
    - when swapping e3:
      - (e1_i, r1_i, e2_i) and (e1_j, r1_j, e2_j) must be protected to avoid causing the edited row become invalid because of a later row edit the 1-hop (entity, relation)
      - the [(e1, r1) and (e2_new, r2)] or (e2, r2) of other rows must be updated with current facts to avoid evaluating rows with stale (entity, relation).
    - when evaluating knowledge editing methods performance:
      - before editing, the model must be able to answer row with row_role "edit":
        - (e1_i, r1_i, e2_i)
        - (e1_j, r1_j, e2_j)
      - the model must learn:
        - (e2_i, r2_i, e3_j)
        - (e2_j, r2_j, e3_i)
      - when evaluating the row with row_role "propagated_eval":
        - the rows with row_role "edit" must be learnt first
        - if "propagated_from" is [("e2_new", a), ("e3_new", b)]
        - if "propagated_from" is [("e3_new", a)], the model must be able to answer (e1, r1, e2) of the row
        - if not, then it's an invalid row
    - swapping_strategy == "e3" does 1 edit. We can analyze whether the model think about e2 and output e3 when asked (e1, r1, r2):
      - (Louis IX of France, mother, Blance of Castile, father, Stanley Armour Dunham)
        - The father of the mother of Louis IX of France is
"""

# %%
# Import necessary libraries and modules

import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, Literal

from tqdm.auto import tqdm

import sys
import logging

from IPython.display import display

# %%

logging.basicConfig(
  level=logging.DEBUG,
  stream=sys.stdout
)

# %%

def parse_args():
  parser = argparse.ArgumentParser(description="Two-hop counterfactual analysis")
  default_input_path = Path(__file__).resolve().parents[1] / "datasets" / "two_hop.csv"
  parser.add_argument(
    "--input_path",
    type=str,
    default=default_input_path,
    help="Path to the two-hop CSV dataset.",
  )
  parser.add_argument(
      "--swapping_strategy",
      type=str,
      default="e2",
      choices=["e2", "e3"],
      help="The swapping strategy to generate counterfactuals, either swapping `e2` values or swapping `e3` values.",
  )
  parser.add_argument(
    "--output_path",
    type=str,
    help="Path to save the generated counterfactuals CSV dataset.",
  )
  args, _ = parser.parse_known_args()
  
  if args.output_path is None:
    args.output_path = Path(__file__).resolve().parents[1] / "datasets" / f"two_hop_cf_{args.swapping_strategy}.csv"
  
  return args

args = parse_args()

# %%

def initialize_stage0_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Stage 0: initialize output columns and row state columns."""
  for col in ["e2_new", "e2_label_new", "e2_aliases_new", "e3_new", "e3_label_new", "e3_aliases_new", "propagated_from"]:
    df[col] = pd.NA

  df["swapped_with"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
  df["row_role"] = "unused"
  return df

def build_group_index(df: pd.DataFrame) -> Dict[Tuple[Any, Any], list[int]]:
  """
  Stage 1 index:
  IDX_GROUP[(r1_type, r2_type)] -> ordered list of row_ids.
  """
  idx_group: Dict[Tuple[Any, Any], list[int]] = defaultdict(list)
  for row_id, row in tqdm(df.iterrows(), total=len(df), desc="Stage 1: build IDX_GROUP"):
    idx_group[(row["r1_type"], row["r2_type"])].append(row_id)
  return dict(idx_group)

def is_valid_swappable_pair(df: pd.DataFrame, row_i: int, row_j: int) -> bool:
  """Apply Stage 1 validity constraints for a direct swap candidate pair."""
  if row_i == row_j:
    return False

  ri = df.loc[row_i]
  rj = df.loc[row_j]

  if ri["e1"] == rj["e1"]:
    return False

  if ri["e1"] == rj["e2"]:
    return False
  if ri["e1"] == rj["e3"]:
    return False
  
  if rj["e1"] == ri["e2"]:
    return False
  if rj["e1"] == ri["e3"]:
    return False

  return True

def _iter_group_candidates_deterministic(
  df: pd.DataFrame,
  row_i: int,
  idx_group: Dict[Tuple[Any, Any], list[int]],
):
  """Yield candidate row ids with deterministic order: >i first, then <=i."""
  row_i_data = df.loc[row_i]
  group_key = (row_i_data["r1_type"], row_i_data["r2_type"])
  candidate_pool = sorted(idx_group.get(group_key, []))

  for row_j in candidate_pool:
    if row_j > row_i:
      yield row_j
  for row_j in candidate_pool:
    if row_j <= row_i:
      yield row_j

def _set_new_target(
  df: pd.DataFrame,
  row_id: int,
  target_col_prefix: Literal["e2", "e3"],
  target_id: Any,
  target_label: str,
  target_aliases: list[str],
) -> None:
  """Write target id + metadata to {e2|e3}_new fields."""
  df.at[row_id, f"{target_col_prefix}_new"] = target_id
  df.at[row_id, f"{target_col_prefix}_label_new"] = target_label
  df.at[row_id, f"{target_col_prefix}_aliases_new"] = target_aliases

def register_kb_rewrite(
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], list[Dict[str, Any]]],
  subject: Any,
  relation: Any,
  target_old: Any,
  target_new: Any,
  target_label_new: Any,
  target_aliases_new: Any,
  target_row_index: int,
  source_row_index: int,
  edit_scope: Literal["hop1", "hop2"],
) -> Dict[str, Any]:
  """
  Enforce Stage 0 KB_MEM rule:
  - one old fact key has at most one canonical target_new
  - if key already exists, reuse existing rewrite (memory-first)
  """
  sr_key = (subject, relation)
  sro_key = (subject, relation, target_old)
  
  if sro_key in kb_mem:
    logging.debug(f"Warning: attempted to register conflicting rewrite for {sro_key} at row {source_row_index}, but key already has rewrite to {kb_mem[sro_key]['target_new']} from row {kb_mem[sro_key]['source_row_index']}. Reusing existing rewrite.")
    return kb_mem[sro_key]
  
  data = {
    "target_new": target_new,
    "target_label_new": target_label_new,
    "target_aliases_new": target_aliases_new,
    "target_row_index": target_row_index,
    "source_row_index": source_row_index,
    "edit_scope": edit_scope,
  }
  
  kb_mem[sro_key] = data

  canonical_entry = kb_mem[sro_key]
  
  if len(kb_mem_sr[sr_key]) > 0:
    logging.debug(f"Warning: multiple rewrites for (subject, relation) pair {sr_key} at row {source_row_index}. This should not happen under current KB_MEM registration rules since each (subject, relation) pair should only have one target_new. Current rewrites for this pair: {kb_mem_sr[sr_key]}.")
  
  kb_mem_sr[(subject, relation)].append({
    "target_old": target_old,
    **data,
  })
  return canonical_entry

def evaluate_row_for_e3(
  df: pd.DataFrame,
  row_id: int,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  protected_facts: Set[Tuple[Any, Any, Any]],
) -> Tuple[Literal["blocked", "propagated", "eligible_for_direct_edit"], Any]:
  """Stage 0 reusable e3 evaluator (for both i and j rows)."""
  row = df.loc[row_id]
  if pd.notna(row["swapped_with"]):
    return "swapped_row"
  
  hop1_key = (row["e1"], row["r1"], row["e2"])
  hop2_key = (row["e2"], row["r2"], row["e3"])

  if hop2_key in protected_facts:
    return "protected_fact"
  
  if hop1_key in kb_mem or hop2_key in kb_mem:
    return "propagated_row"

  return "eligible_for_direct_edit"

def evaluate_row_for_e2(
  df: pd.DataFrame,
  row_id: int,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  protected_facts: Set[Tuple[Any, Any, Any]],
) -> Tuple[Literal["blocked", "propagated", "eligible_for_direct_edit"], Any]:
  """Stage 0 reusable e2 evaluator (for both i and j rows)."""
  row = df.loc[row_id]

  if pd.notna(row["swapped_with"]):
    return "swapped_row"
  
  hop1_key = (row["e1"], row["r1"], row["e2"])
  hop2_key = (row["e2"], row["r2"], row["e3"])
  
  if hop1_key in protected_facts:
    return "protected_fact"
  
  if hop1_key in kb_mem or hop2_key in kb_mem:
    return "propagated_row"
  
  return "eligible_for_direct_edit"

def apply_stage2a_e2(
  df: pd.DataFrame,
  idx_group: Dict[Tuple[Any, Any], list[int]],
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  protected_facts: Set[Tuple[Any, Any, Any]],
) -> Dict[str, int]:
  """
  Stage 2A: apply counterfactuals for swapping_strategy == 'e2'.
  """
  stats = {
    "processed": 0,
    "swapped_row": 0,
    "protected_fact": 0,
    "propagated_row": 0,
    "eligible_for_direct_edit": 0,
    "no_partner": 0,
    "edited_pairs": 0,
    "edited_rows": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_i in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 2A (e2): apply swaps"):
    stats["processed"] += 1

    state_i = evaluate_row_for_e2(df, row_i, kb_mem, protected_facts)
    if state_i == "swapped_row":
      stats["swapped_row"] += 1
      continue
    elif state_i == "protected_fact":
      stats["protected_fact"] += 1
      continue
    elif state_i == "propagated_row":
      stats["propagated_row"] += 1
      continue
    elif state_i != "eligible_for_direct_edit":
      logging.debug(f"Warning: unexpected state_i {state_i} for row {row_i}")
      continue
    
    stats["eligible_for_direct_edit"] += 1

    row_i_data = df.loc[row_i]

    selected_j_data = None

    for row_j in _iter_group_candidates_deterministic(df, row_i, idx_group):
      if row_j == row_i:
        continue
      if pd.notna(df.at[row_j, "swapped_with"]):
        continue
      if not is_valid_swappable_pair(df, row_i, row_j):
        continue

      row_j_data = df.loc[row_j]

      if row_i_data["e2"] == row_j_data["e2"]:
        continue
        
      if ((row_j_data["e2"], row_i_data["r2"], row_i_data["e3"]) in kb_mem) or ((row_i_data["e2"], row_j_data["r2"], row_j_data["e3"]) in kb_mem):
        continue
      
      state_j = evaluate_row_for_e2(df, row_j, kb_mem, protected_facts)
      if state_j != "eligible_for_direct_edit":
        continue

      selected_j_data = row_j_data
      break

    if selected_j_data is None:
      stats["no_partner"] += 1
      continue

    _set_new_target(
      df,
      row_i,
      "e2",
      selected_j_data["e2"],
      selected_j_data["e2_label"],
      selected_j_data["e2_aliases"],
    )
    _set_new_target(
      df,
      selected_j_data["id"],
      "e2",
      row_i_data["e2"],
      row_i_data["e2_label"],
      row_i_data["e2_aliases"],
    )

    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      row_i_data["e1"],
      row_i_data["r1"],
      row_i_data["e2"],
      selected_j_data["e2"],
      selected_j_data["e2_label"],
      selected_j_data["e2_aliases"],
      row_i,
      selected_j_data["id"],
      "hop1",
    )
    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      selected_j_data["e1"],
      selected_j_data["r1"],
      selected_j_data["e2"],
      row_i_data["e2"],
      row_i_data["e2_label"],
      row_i_data["e2_aliases"],
      selected_j_data["id"],
      row_i,
      "hop1",
    )

    df.at[row_i, "row_role"] = "edit"
    df.at[row_i, "swapped_with"] = selected_j_data["id"]
    
    df.at[selected_j_data["id"], "row_role"] = "edit"
    df.at[selected_j_data["id"], "swapped_with"] = row_i

    if row_i_data["e3"] != selected_j_data["e3"]:
      _set_new_target(
        df,
        row_i,
        "e3",
        selected_j_data["e3"],
        selected_j_data["e3_label"],
        selected_j_data["e3_aliases"],
      )

      _set_new_target(
        df,
        selected_j_data["id"],
        "e3",
        row_i_data["e3"],
        row_i_data["e3_label"],
        row_i_data["e3_aliases"],
      )
    
    protected_facts.add((selected_j_data["e2"], row_i_data["r2"], selected_j_data["e3"]))
    protected_facts.add((row_i_data["e2"], selected_j_data["r2"], row_i_data["e3"]))

    stats["edited_pairs"] += 1
    stats["edited_rows"] += 2

  return stats

def apply_stage2b_e3(
  df: pd.DataFrame,
  idx_group: Dict[Tuple[Any, Any], list[int]],
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  protected_facts: Set[Tuple[Any, Any, Any]],
) -> Dict[str, int]:
  """
  Stage 2B: apply counterfactuals for swapping_strategy == 'e3'.
  """
  stats = {
    "processed": 0,
    "swapped_row": 0,
    "protected_fact": 0,
    "propagated_row": 0,
    "eligible_for_direct_edit": 0,
    "no_partner": 0,
    "edited_pairs": 0,
    "edited_rows": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_i in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 2B (e3): apply swaps"):
    stats["processed"] += 1

    state_i = evaluate_row_for_e3(df, row_i, kb_mem, protected_facts)
    if state_i == "swapped_row":
      stats["swapped_row"] += 1
      continue
    elif state_i == "protected_fact":
      stats["protected_fact"] += 1
      continue
    elif state_i == "propagated_row":
      stats["propagated_row"] += 1
      continue
    elif state_i != "eligible_for_direct_edit":
      logging.debug(f"Warning: unexpected state_i {state_i} for row {row_i}")
      continue
    
    stats["eligible_for_direct_edit"] += 1

    row_i_data = df.loc[row_i]
    
    selected_j_data: Optional[int] = None

    for row_j in _iter_group_candidates_deterministic(df, row_i, idx_group):
      if row_j == row_i:
        continue
      if pd.notna(df.at[row_j, "swapped_with"]):
        continue
      if not is_valid_swappable_pair(df, row_i, row_j):
        continue

      row_j_data = df.loc[row_j]
      
      if row_i_data["e3"] == row_j_data["e3"]:
        continue

      state_j = evaluate_row_for_e3(df, row_j, kb_mem, protected_facts)
      if state_j != "eligible_for_direct_edit":
        continue

      selected_j_data = row_j_data
      break

    if selected_j_data is None:
      stats["no_partner"] += 1
      continue

    _set_new_target(
      df,
      row_i,
      "e3",
      selected_j_data["e3"],
      selected_j_data["e3_label"],
      selected_j_data["e3_aliases"],
    )
    _set_new_target(
      df,
      selected_j_data["id"],
      "e3",
      row_i_data["e3"],
      row_i_data["e3_label"],
      row_i_data["e3_aliases"],
    )

    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      row_i_data["e2"],
      row_i_data["r2"],
      row_i_data["e3"],
      selected_j_data["e3"],
      selected_j_data["e3_label"],
      selected_j_data["e3_aliases"],
      row_i,
      selected_j_data["id"],
      "hop2",
    )
    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      selected_j_data["e2"],
      selected_j_data["r2"],
      selected_j_data["e3"],
      row_i_data["e3"],
      row_i_data["e3_label"],
      row_i_data["e3_aliases"],
      selected_j_data["id"],
      row_i,
      "hop2",
    )

    df.at[row_i, "row_role"] = "edit"
    df.at[row_i, "swapped_with"] = selected_j_data["id"]
    
    df.at[selected_j_data["id"], "row_role"] = "edit"
    df.at[selected_j_data["id"], "swapped_with"] = row_i

    protected_facts.add((row_i_data["e1"], row_i_data["r1"], row_i_data["e2"]))
    protected_facts.add((selected_j_data["e1"], selected_j_data["r1"], selected_j_data["e2"]))

    stats["edited_pairs"] += 1
    stats["edited_rows"] += 2

  return stats

def apply_stage3_violation_detection(
  df: pd.DataFrame,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  swapping_strategy: Literal["e2", "e3"],
) -> Dict[str, int]:
  stats = {
    "processed": 0,
    "non_edit_row": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_id in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 3: resolve conflicts"):
    stats["processed"] += 1
    
    row_data = df.loc[row_id]
    
    if row_data["row_role"] != "edit":
      stats["non_edit_row"] += 1
      continue
      
    # swapping_strategy == "e2" guarantees hop-2 are not edited
    if swapping_strategy == "e2":
      if row_data["e2_new"] is pd.NA:
        logging.debug(f"Warning: row {row_id} has row_role 'edit' but e2_new is NA. This should not happen under e2 swapping strategy.")
        break
      
      if row_data["e2_new"] == row_data["e2"]:
        logging.debug(f"Warning: row {row_id} has row_role 'edit' but e2_new is same as original e2. This should not happen under e2 swapping strategy.")
        break
      
      hop2_key1 = (row_data["e2_new"], row_data["r2"], row_data["e3"])
      if hop2_key1 in kb_mem:
        logging.debug(f"Conflict detected for row {row_id} on hop2_key {hop2_key1} from memory. This should not happen under e2 swapping strategy since e3 should not be edited.")
        break
      
      hop2_key2 = (row_data["e2_new"], row_data["r2"], row_data["e3_new"])
      if hop2_key2 in kb_mem:
        logging.debug(f"Conflict detected for row {row_id} on hop2_key {hop2_key2} from memory. This should not happen under e2 swapping strategy since e3 should not be edited.")
        break
    
    # swapping_strategy == "e3" guarantees hop-1 are not edited
    if swapping_strategy == "e3":
      if row_data["e2_new"] is not pd.NA:
        logging.debug(f"Warning: row {row_id} has row_role 'edit' and non-NA e2_new. This should not happen under e3 swapping strategy since e2 should not be edited.")
        break
      
      if row_data["e3_new"] == row_data["e3"]:
        logging.debug(f"Warning: row {row_id} has row_role 'edit' but e3_new is same as original e3. This should not happen under e3 swapping strategy.")
        break
      
      hop1_key = (row_data["e1"], row_data["r1"], row_data["e2"])
      if hop1_key in kb_mem:
        logging.debug(f"Conflict detected for row {row_id} on hop1_key {hop1_key} from memory. This should not happen under e3 swapping strategy since e2 should not be edited.")
        break

      hop2_key = (row_data["e2"], row_data["r2"], row_data["e3_new"])
      if hop2_key in kb_mem:
        logging.debug(f"Conflict detected for row {row_id} on hop2_key {hop2_key} from memory. This should not happen under e3 swapping strategy since e2 should not be edited.")
        break
    
    continue

  return stats

def apply_stage4_propagation(
  df: pd.DataFrame,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
) -> Dict[str, int]:
  """
  Stage 4: propagate edits to create "propagated_eval" rows.
  """
  stats = {
    "processed": 0,
    "edit_row": 0,
    "no_change": 0,
    "hop1_propagated_hop2_no_target": 0,
    "hop1_hop2_propagated": 0,
    "hop2_propagated": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_id in tqdm(sorted_row_ids, total=len(df), desc="Stage 4: propagate edits"):
    stats["processed"] += 1

    row_data = df.loc[row_id]
    if row_data["row_role"] == "edit":
      stats["edit_row"] += 1
      continue
    
    hop1_key = (row_data["e1"], row_data["r1"], row_data["e2"])
    
    if hop1_key in kb_mem:
      hop1_entry = kb_mem[hop1_key]
      
      hop2_key = (hop1_entry["target_new"], row_data["r2"])
      
      if hop2_key not in kb_mem_sr:
        stats["hop1_propagated_hop2_no_target"] += 1
        continue
      
      hop2_entries = kb_mem_sr[hop2_key]
      if len(hop2_entries) > 1:
        logging.debug(f"Warning: multiple hop2 rewrites found for hop2_key {hop2_key} during propagation of row {row_id}. This should not happen under current KB_MEM registration rules.")
        break
      hop2_entry = hop2_entries[0]
      
      if hop1_entry["target_new"] != row_data["e2"]:
        _set_new_target(
          df,
          row_id,
          "e2",
          hop1_entry["target_new"],
          hop1_entry["target_label_new"],
          hop1_entry["target_aliases_new"],
        )
      if hop2_entry["target_new"] != row_data["e3"]:
        _set_new_target(
          df,
          row_id,
          "e3",
          hop2_entry["target_new"],
          hop2_entry["target_label_new"],
          hop2_entry["target_aliases_new"],
        )
      
      df.at[row_id, "row_role"] = "propagated_eval"
      df.at[row_id, "propagated_from"] = [("e2_new", hop1_entry["source_row_index"]), ("e3_new", hop2_entry["source_row_index"])]
      
      stats["hop1_hop2_propagated"] += 1
    else:
      hop2_key = (row_data["e2"], row_data["r2"], row_data["e3"])
      if hop2_key not in kb_mem:
        stats["no_change"] += 1
        continue
      
      hop2_entry = kb_mem[hop2_key]
      
      if hop2_entry["target_new"] != row_data["e3"]:
        _set_new_target(
          df,
          row_id,
          "e3",
          hop2_entry["target_new"],
          hop2_entry["target_label_new"],
          hop2_entry["target_aliases_new"],
        )
      
      df.at[row_id, "row_role"] = "propagated_eval"
      df.at[row_id, "propagated_from"] = [("e3_new", hop2_entry["source_row_index"])]

      stats["hop2_propagated"] += 1

  return stats

def apply_stage5_sorting(df: pd.DataFrame) -> pd.DataFrame:
  """
  Stage 5: sort rows for export.
  - guarantee `propagated_eval` rows are after `edit` rows, and `unused` rows are at the end
  - within `propagated_eval`, prefer rows that reference earlier source edits so propagated rows follow their sources
  - stable and deterministic: falls back to original row index when ties occur

  Returns the sorted DataFrame (does not save to disk).
  """
  # Build interleaved ordering: place each `edit` then any propagated rows
  # whose dependencies are fully satisfied (all source row ids already placed).
  orig_index = list(df.index)

  edit_ids = [i for i in orig_index if df.at[i, "row_role"] == "edit"]
  propagated_ids = [i for i in orig_index if df.at[i, "row_role"] == "propagated_eval"]

  dep_map = {p: set([t[1] for t in df.at[p, "propagated_from"]]) for p in propagated_ids}

  placed = set()
  out_order: list[int] = []

  remaining_propagated_ids = set(propagated_ids)

  for edit_id in edit_ids:
    out_order.append(edit_id)
    placed.add(edit_id)

    # greedily place any propagated rows unlocked by this placement
    for propagated_id in sorted(list(remaining_propagated_ids)):
      if dep_map[propagated_id].issubset(placed):
        out_order.append(propagated_id)
        placed.add(propagated_id)
        remaining_propagated_ids.remove(propagated_id)
  
  # finally append unused rows
  for i in orig_index:
    if i in placed:
      continue
    
    if df.at[i, "row_role"] != "unused":
      logging.debug(f"Warning: active non-edit row {i} with row_role {df.at[i, 'row_role']} was not placed after interleaving edits and propagated rows. Placing it now. This should not happen under current stage 2 and 4 logic since all active rows should be either edits or propagated, but we include this as a safety catch-all.")
      break
    
    out_order.append(i)
    
  if len(df) != len(out_order):
    logging.debug(f"Warning: output order length {len(out_order)} does not match dataframe length {len(df)}. This should not happen. Falling back to original order.")
    return df.copy()

  df_sorted = df.loc[out_order].copy()
  return df_sorted

def apply_stage6_export(df: pd.DataFrame, output_path: str | Path) -> Dict[str, Any]:
  """
  Stage 6: final benchmark export.
  """
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  # remove unused rows in-place so downstream code operates only on active rows
  inactive_df = df[df["row_role"] == "unused"]
  if len(inactive_df) > 0:
    df.drop(index=inactive_df.index, inplace=True)

  edit_df = df[df["row_role"] == "edit"]
  propagated_df = df[df["row_role"] == "propagated_eval"]
  # after dropping, active_df is the current dataframe
  active_df = df

  active_df.to_csv(output_path, index=False)

  return {
    "output_path": str(output_path),
    "edit_rows": len(edit_df),
    "propagated_rows": len(propagated_df),
    "active_rows": len(active_df),
    "unused_rows": len(inactive_df),
  }

# %%
# Load the two-hop dataset

df = pd.read_csv(args.input_path)

t1 = df[["e1", "r1", "e2"]].rename(columns={"e1": "entity", "r1": "relation", "e2": "target"})
t2 = df[["e2", "r2", "e3"]].rename(columns={"e2": "entity", "r2": "relation", "e3": "target"})
all_triples = pd.concat([t1, t2])
target_counts = all_triples.groupby(["entity", "relation"])["target"].nunique()
valid_pairs = target_counts[target_counts == 1].index

idx1 = pd.MultiIndex.from_arrays([df["e1"], df["r1"]])
idx2 = pd.MultiIndex.from_arrays([df["e2"], df["r2"]])

df = df[idx1.isin(valid_pairs) & idx2.isin(valid_pairs)]
relations = ["mother", "father", "present in work", "notable work", "location of formation", "founded by", "country of origin"]
mask_non_stale_relations = df["r1_type"].isin(relations) & df["r2_type"].isin(relations)
df = df[mask_non_stale_relations]

# %%

# Stage 0: initialize runtime state
df = initialize_stage0_columns(df)

KB_MEM: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
KB_MEM_SR: Dict[Tuple[Any, Any], list[Dict[str, Any]]] = defaultdict(list)
PROTECTED_FACTS: Set[Tuple[Any, Any, Any]] = set()

print(
  f"Stage 0 complete | rows={len(df)}"
)

# Stage 1: deterministic pairing scaffolding
IDX_GROUP = build_group_index(df)
print(f"Stage 1 complete | groups={len(IDX_GROUP)} | partner_search=ready")

# Stage 2A / 2B (partial): strategy-dependent editing pass
if args.swapping_strategy == "e2":
  stage2a_stats = apply_stage2a_e2(
    df,
    IDX_GROUP,
    KB_MEM,
    KB_MEM_SR,
    PROTECTED_FACTS,
  )
  output = " | ".join([f"{k}={v}" for k, v in stage2a_stats.items()])
  print(f"Stage 2A complete | {output}")
  cols = ["e1_label", "r1_type", "e2_label", "e2_label_new", "r2_type", "e3_label", "e3_label_new", "swapped_with"]
  display(df.loc[[3, 140, 215, 390]][cols])
else:
  stage2b_stats = apply_stage2b_e3(
    df,
    IDX_GROUP,
    KB_MEM,
    KB_MEM_SR,
    PROTECTED_FACTS,
  )
  output = " | ".join([f"{k}={v}" for k, v in stage2b_stats.items()])
  print(f"Stage 2B complete | {output}")
  
# %%

stage3_stats = apply_stage3_violation_detection(
  df,
  KB_MEM,
  args.swapping_strategy,
)
output = " | ".join([f"{k}={v}" for k, v in stage3_stats.items()])
print(f"Stage 3 complete | {output}")

# %%

stage4_stats = apply_stage4_propagation(
  df,
  KB_MEM,
  KB_MEM_SR,
)
output = " | ".join([f"{k}={v}" for k, v in stage4_stats.items()])
print(f"Stage 4 complete | {output}")

# %%
# Stage 5: sort rows for export so propagated rows follow edits and unused are last

df = apply_stage5_sorting(df)
print(f"Stage 5 complete")

cols = ["id", "row_role", "propagated_from"]
df[cols].head(20)

# %%

stage6_stats = apply_stage6_export(df, args.output_path)
output = " | ".join([f"{k}={v}" for k, v in stage6_stats.items()])
print(f"Stage 6 complete | {output}")

# %%

def analysis_summary(df):
  print("="*60)
  print("📊 DATASET SUMMARY")
  print("="*60)

  print(f"Result rows: {len(df)}")
  print()

  print("---- Counterfactual Coverage ----")
  print(f"Rows with e2_new: {df['e2_new'].notna().sum()}")
  print(f"Rows without e2_new: {df['e2_new'].isna().sum()}")

  print(f"Rows with e3_new: {df['e3_new'].notna().sum()}")
  print(f"Rows without e3_new: {df['e3_new'].isna().sum()}")
  print()

  print("---- Row Type Breakdown ----")
  print(df["row_role"].value_counts(dropna=False))
  print()

  print("---- Uniqueness ----")
  print(f"Unique e2: {df['e2'].nunique()}")
  print(f"Unique e2_new: {df['e2_new'].nunique()}")

  print(f"Unique e3: {df['e3'].nunique()}")
  print(f"Unique e3_new: {df['e3_new'].nunique()}")
  print()

  print("="*60)

analysis_summary(df)

# %%
# check overlap

mask = df["e2_new"].notna()
inconsistent = df.loc[mask & (df["e2_new"] == df["e2"])]
print("e2_new overlap:", len(inconsistent))

cols = ["e1_label", "r1_type", "e2_label", "e2_label_new", "r2_type", "e3_label", "e3_label_new"]
inconsistent[cols].head()

# %%

mask = df["e3_new"].notna()
inconsistent = df.loc[mask & (df["e3_new"] == df["e3"])]
print("e3_new overlap:", len(inconsistent))

cols = ["e1_label", "r1_type", "e2_label", "r2_type", "e3_label", "e3_label_new", "row_role", "swapped_with", "propagated_from"]
inconsistent[cols].head()

# %%

def test_knowledge_consistency(df):
  """check if there are multiple e2_new values for the same (e1, r1, e2) group"""
  group_check = df.groupby(["e1","r1","e2"])["e2_new"].nunique()
  return (group_check > 1).sum() == 0

print(f"knowledge consistent: {test_knowledge_consistency(df)}")

# %%

def count_consecutive_edit_rows(df):
  """
  Count how many edit rows are immediately followed by another edit row.
  """

  # reset index to ensure sequential access
  df_seq = df.reset_index(drop=True)

  count = 0
  indices = []

  for i in range(len(df_seq) - 1):
    if df_seq.loc[i, "row_role"] == "edit" and df_seq.loc[i+1, "row_role"] == "edit":
      count += 1
      indices.append(i)

  return count

print(f"Consecutive edit rows: {count_consecutive_edit_rows(df)}")

# %%