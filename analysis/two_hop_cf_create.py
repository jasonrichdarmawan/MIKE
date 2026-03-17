"""
---
description: Clear pseudo algorithm for generating two-hop counterfactuals with conflict checks.

Notation:
- A row i has triplets:
  - hop-1: (e1_i, r1_i, e2_i)
  - hop-2: (e2_i, r2_i, e3_i)
- Hop key definitions (explicit):
  - hop-1 key is only (e1_i, r1_i)
  - hop-2 key is only (e2_i, r2_i)
  - both keys must be checked during resolution
- A candidate partner row j is in the same template group:
  - r1_type_i == r1_type_j and r2_type_i == r2_type_j

Validity constraints for a swappable pair (i, j):
- i != j
- e1_i != e1_j
- no identity leakage after swap:
  - e1_i != e2_j
  - e1_j != e2_i
  - e1_i != e3_j
  - e1_j != e3_i

Stage 0: Initialize output columns
- e2_new, e2_label_new, e2_aliases_new
- e3_new, e3_label_new, e3_aliases_new
- swapped_with (partner row index)
- row_role in {"edit", "propagated_eval", "unused"}
- Initialize memory bank `KB_MEM` (fact rewrite map, not row storage):
  - key: (subject_id, relation_id, target_old_id)
  - value: {
      target_new,
      target_label_new,
      target_aliases_new,
      source_row_index,
      edit_scope in {"hop1", "hop2"}
    }
- Interpretation of `KB_MEM`:
  - each entry means: "rewrite this exact old fact to this new target"
  - it does NOT represent a full row and does NOT directly encode both hops of a row
- `KB_MEM` rule:
  - one old fact key has at most one canonical `target_new`
  - different keys can coexist for multi-fact relations, e.g.:
    - (Google, founder, Sergey Brin) -> Jeff Bezos
    - (Google, founder, Larry Page) -> Larry Page (unchanged / no rewrite)
  - if a key already exists, always reuse stored `target_new` (memory-first), do not create a second swap for that key
  - maintain an auxiliary lookup view `KB_MEM_SR[(subject_id, relation_id)] -> set(target_new)`
    for fast hop-level resolution before using dataset fallback
- Initialize protected facts set `PROTECTED_FACTS` (used only for swapping_strategy == "e3"):
  - stores exact facts `(e1, r1, e2)` that must never be edited
  - purpose: prevent silent conflicts where a later direct e3 edit would rewrite a fact used as a protected anchor
- Reusable helper for e3 flow:
  - `resolve_effective_subject(row_x)`:
    - if `(e1_x, r1_x, e2_x)` in `KB_MEM`, set `e2_new_x` from memory and return `e2_eff_x = e2_new_x`
    - else return `e2_eff_x = e2_x`
  - `evaluate_row_for_e3(row_x)` (used for both i and j):
    - input:
      - row_x with current fields (`e1_x, r1_x, e2_x, r2_x, e3_x, swapped_with_x, row_role_x`)
      - global stores (`KB_MEM`, `PROTECTED_FACTS`)
    - output:
      - `state` in {`blocked`, `propagated`, `eligible_for_direct_edit`}
      - `e2_eff_x`
    - exact logic (same for any row_x):
      1) if `swapped_with_x` is not null:
         - return `state = blocked`
      2) set `e2_eff_x = resolve_effective_subject(row_x)`
      3) if `(e2_eff_x, r2_x, e3_x)` in `PROTECTED_FACTS`:
         - this row cannot be directly edited in e3
         - if `row_role_x` is not `edit`, set `row_role_x = propagated_eval`
         - return `state = propagated`
      4) if `(e2_eff_x, r2_x, e3_x)` in `KB_MEM`:
         - set `e3_new_x` from exact memory entry
         - if `row_role_x` is not `edit`, set `row_role_x = propagated_eval`
         - return `state = propagated`
      5) otherwise:
         - return `state = eligible_for_direct_edit`
- Reusable helper for e2 flow:
  - `evaluate_row_for_e2(row_x)` (used for both i and j):
    - returns:
      - `state` in {`blocked`, `propagated`, `eligible_for_direct_edit`}
      - `e2_eff_x`
    - exact logic (same for any row_x):
      1) if `swapped_with_x` is not null:
         - return `state = blocked`
      2) if `(e1_x, r1_x, e2_x)` in `KB_MEM`:
         - set `e2_new_x` from memory
         - set `row_role_x = propagated_eval` (if not edit)
         - set `e2_eff_x = e2_new_x`
         - resolve `e3_new_x` from `KB_MEM_SR[(e2_eff_x, r2_x)]`, else smallest row_id in `HOP_INDEX[(e2_eff_x, r2_x)]`
         - if `e3_new_x` is None:
           - set `row_role` to `unused` if not already `edit`
           - return `state = blocked`
         - else:
           - return `state = propagated`
      3) else:
         - set `e2_eff_x = e2_x`
         - return `state = eligible_for_direct_edit`

- Build unified hop lookup index (row-id based):
  - `HOP_INDEX[(subject, relation)] -> ordered list of row_ids`
  - iterate each row once and insert two items:
    - add row_id to key (e1, r1)
    - add row_id to key (e2, r2)
  - index is immutable (built once from original dataset)
  - fallback lookup returns row ids; selected target is decided by which hop matches the queried key

Stage 1: Build deterministic pairings inside each (r1_type, r2_type) group
- Build indices once (avoid O(N^2) scans):
  - IDX_GROUP[(r1_type, r2_type)] -> row ids
  - HOP_INDEX[(subject, relation)] -> candidate row ids
- Partner search is row-wise with index lookup:
  - for row i, retrieve candidate pool from index, then filter by validity constraints
  - exclude rows already swapped (`swapped_with` is not null) from candidate pool
  - choose deterministic best partner (e.g., smallest valid row id > i, else smallest valid row id)

Stage 2A: Apply counterfactuals when swapping_strategy == "e2"
- Process rows in ascending index i:
  1) run `evaluate_row_for_e2(row_i)`:
    - if state is `blocked`, skip row i
    - if state is `propagated`, continue (helper already resolved `e3_new_i`)
  2) find partner candidates from IDX_GROUP[(r1_type_i, r2_type_i)] with validity constraints
  3) for each candidate j in deterministic order, run `evaluate_row_for_e2(row_j)` (same rules):
    - choose first j where state is `eligible_for_direct_edit`
  4) if no eligible j found:
    - keep row i as propagated/unused path; do not force `edit`
  5) if eligible j is found, perform symmetric e2 swap edit:
    - recompute hop-2 for both sides:
     - `e3_new_i` from `KB_MEM_SR[(e2_new_i, r2_i)]` else smallest row_id in `HOP_INDEX[(e2_new_i, r2_i)]`
     - `e3_new_j` from `KB_MEM_SR[(e2_new_j, r2_j)]` else smallest row_id in `HOP_INDEX[(e2_new_j, r2_j)]`
     - if either `e3_new_i` or `e3_new_j` is None, this pair fails hop-2 resolution and cannot be directly edited; skip to next candidate j
    - set `e2_new_i <- e2_j` and `e2_new_j <- e2_i`
    - write both hop-1 rewrites to `KB_MEM`:
     - `KB_MEM[(e1_i, r1_i, e2_i)] = e2_new_i`
     - `KB_MEM[(e1_j, r1_j, e2_j)] = e2_new_j`
    - set `row_role_i = "edit"`, `row_role_j = "edit"`
    - set `swapped_with_i = j` and `swapped_with_j = i`

Stage 2B: Apply counterfactuals when swapping_strategy == "e3"
- Process rows in ascending index i:
  1) run `evaluate_row_for_e3(row_i)`:
    - if state is `blocked`, skip row i
    - if state is `propagated`, continue to next i
  2) find partner candidates from IDX_GROUP[(r1_type_i, r2_type_i)] with validity constraints
  3) for each candidate j in deterministic order, run `evaluate_row_for_e3(row_j)` (same rules):
    - choose first j where state is `eligible_for_direct_edit`
  4) if no eligible j found:
    - keep row i as propagated/unused path; do not force `edit`
  5) if eligible j is found, perform symmetric e3 swap edit:
    - set e3_new_i <- e3_j and e3_new_j <- e3_i
    - write both exact hop-2 rewrites to `KB_MEM`:
      - `KB_MEM[(e2_eff_i, r2_i, e3_i)] = e3_new_i`
      - `KB_MEM[(e2_eff_j, r2_j, e3_j)] = e3_new_j`
    - set `row_role_i = "edit"`, `row_role_j = "edit"`
    - set `swapped_with_i = j` and `swapped_with_j = i`
    - register protected facts anchors:
      - add `(e1_i, r1_i, e2_i)` to `PROTECTED_FACTS`
      - add `(e1_j, r1_j, e2_j)` to `PROTECTED_FACTS` for selected partner j

Stage 3: Resolve knowledge conflict from memory bank (strategy-agnostic)
- Process each row i in ascending index:
  0) Initialize per-row exclusion control:
    - `set_unused = False`
  1) Hop-1 propagation check (exact triple):
    - if `(e1, r1, e2)` exists in `KB_MEM`, read `target_new = KB_MEM[(e1, r1, e2)]`
    - if `target_new != e2`:
      - set `e2_new = target_new`
      - if `row_role` is `unused`, set `row_role = propagated_eval`
  2) Define effective subject:
    - `e2_eff = e2_new` if present else `e2`
  3) if `e2_eff != e2`:
    - if `(e2_eff, r2)` in `KB_MEM_SR`, set `e3_new` from memory
    - else if `HOP_INDEX[(e2_eff, r2)]` has at least one row_id:
      - set `e3_new` from smallest row_id in `HOP_INDEX[(e2_eff, r2)]`
    - else:
      - if `row_role == propagated_eval`, set `set_unused = True`
  4) if `(e2_eff, r2, e3)` exists in `KB_MEM`:
    - if `row_role == edit` and 
    - set `e3_new` from memory
    - if `row_role == unused`, set `row_role = propagated_eval`
  5) Branch outcome rule:
    - if `set_unused == True`, set `row_role = unused`
    - else keep current role (`propagated_eval` or `unused`)

Why recompute `e3_new` after an `e2` edit:
- editing (e1, r1, e2_old) -> e2_new changes the subject of hop-2 queries from e2_old to e2_new.
- therefore `e3_new` should be derived from the updated key (e2_new, r2), not kept from the original pair placeholder.

Stage 4: Final benchmark export
- Keep full table with `row_role` for traceability.
- Editing benchmark uses rows with row_role == "edit".
- Generalization-only evaluation uses rows with row_role == "propagated_eval".
- Exclude row_role == "unused".

Important clarification on notation:
- Comparisons must always include row ids, e.g. e2_i == e2_j or e3_i == e3_j.
- Expressions like "e2 == e2" are shorthand and should not be used in executable logic.
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
# Load the two-hop dataset

df = pd.read_csv(args.input_path)

# %%

def initialize_stage0_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Stage 0: initialize output columns and row state columns."""
  for col in ["e2_new", "e2_label_new", "e2_aliases_new", "e3_new", "e3_label_new", "e3_aliases_new"]:
    df[col] = pd.NA

  df["swapped_with"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
  df["row_role"] = "unused"
  return df

def build_entity_lookup(df: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
  """Create entity_id -> {label, aliases} lookup from e1/e2/e3 columns."""
  entity_lookup: Dict[Any, Dict[str, Any]] = {}
  entity_specs = [
    ("e1", "e1_label", "e1_aliases"),
    ("e2", "e2_label", "e2_aliases"),
    ("e3", "e3_label", "e3_aliases"),
  ]

  for row in tqdm(df.itertuples(index=False), total=len(df), desc="Stage 0: build entity lookup"):
    for entity_col, label_col, aliases_col in entity_specs:
      entity_id = getattr(row, entity_col)
      if pd.isna(entity_id) or entity_id in entity_lookup:
        continue
      entity_lookup[entity_id] = {
        "label": getattr(row, label_col),
        "aliases": getattr(row, aliases_col),
      }

  return entity_lookup

def build_hop_index(df: pd.DataFrame) -> Dict[Tuple[Any, Any], list[int]]:
  """
  Stage 0: build immutable row-id based index.
  HOP_INDEX[(subject, relation)] -> ordered list of row_ids,
  inserting both (e1, r1) and (e2, r2) for every row.
  """
  hop_index: Dict[Tuple[Any, Any], list[int]] = defaultdict(list)
  for row_id, row in tqdm(df.iterrows(), total=len(df), desc="Stage 0: build HOP_INDEX"):
    hop_index[(row["e1"], row["r1"])].append(row_id)
    hop_index[(row["e2"], row["r2"])].append(row_id)

  return dict(hop_index)

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
  if rj["e1"] == ri["e2"]:
    return False

  if ri["e1"] == rj["e3"]:
    return False
  if rj["e1"] == ri["e3"]:
    return False

  return True

def _deterministic_partner_sort_key(anchor_row_id: int, candidate_row_id: int) -> Tuple[int, int]:
  """
  Deterministic preference:
  1) smallest valid row_id > i
  2) otherwise smallest valid row_id
  """
  return (0 if candidate_row_id > anchor_row_id else 1, candidate_row_id)

def find_best_partner_for_row(
  df: pd.DataFrame,
  row_id: int,
  idx_group: Dict[Tuple[Any, Any], list[int]],
  reserved_rows: Optional[Set[int]] = None,
) -> Optional[int]:
  """
  Stage 1 partner search:
  - candidate pool from IDX_GROUP[(r1_type, r2_type)]
  - exclude rows already swapped and reserved rows
  - filter with validity constraints
  - deterministic best-partner rule
  """
  if reserved_rows is None:
    reserved_rows = set()

  row = df.loc[row_id]
  group_key = (row["r1_type"], row["r2_type"])
  candidate_pool = idx_group.get(group_key, [])

  valid_candidates: list[int] = []
  for candidate_row_id in candidate_pool:
    if candidate_row_id == row_id:
      continue
    if candidate_row_id in reserved_rows:
      continue
    if pd.notna(df.at[candidate_row_id, "swapped_with"]):
      continue
    if not is_valid_swappable_pair(df, row_id, candidate_row_id):
      continue
    valid_candidates.append(candidate_row_id)

  if not valid_candidates:
    return None

  return min(valid_candidates, key=lambda c: _deterministic_partner_sort_key(row_id, c))

def build_stage1_pair_plan(
  df: pd.DataFrame,
  idx_group: Dict[Tuple[Any, Any], list[int]],
) -> Dict[int, int]:
  """
  Build deterministic pairings across rows without mutating row roles.
  Returns symmetric mapping: pair_plan[i] = j and pair_plan[j] = i.
  """
  pair_plan: Dict[int, int] = {}
  reserved_rows: Set[int] = set()

  sorted_row_ids = sorted(df.index.tolist())
  for row_id in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 1: pair search"):
    if row_id in reserved_rows:
      continue
    if pd.notna(df.at[row_id, "swapped_with"]):
      continue

    partner_row_id = find_best_partner_for_row(df, row_id, idx_group, reserved_rows=reserved_rows)
    if partner_row_id is None:
      continue

    pair_plan[row_id] = partner_row_id
    pair_plan[partner_row_id] = row_id
    reserved_rows.add(row_id)
    reserved_rows.add(partner_row_id)

  return pair_plan

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
  entity_lookup: Dict[Any, Dict[str, Any]],
  target_label: Optional[Any] = None,
  target_aliases: Optional[Any] = None,
) -> None:
  """Write target id + metadata to {e2|e3}_new fields."""
  df.at[row_id, f"{target_col_prefix}_new"] = target_id

  if target_label is None or pd.isna(target_label):
    target_label = entity_lookup.get(target_id, {}).get("label", pd.NA)
  if target_aliases is None or pd.isna(target_aliases):
    target_aliases = entity_lookup.get(target_id, {}).get("aliases", pd.NA)

  df.at[row_id, f"{target_col_prefix}_label_new"] = target_label
  df.at[row_id, f"{target_col_prefix}_aliases_new"] = target_aliases

def register_kb_rewrite(
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  subject: Any,
  relation: Any,
  target_old: Any,
  target_new: Any,
  target_label_new: Any,
  target_aliases_new: Any,
  source_row_index: int,
  edit_scope: Literal["hop1", "hop2"],
) -> Dict[str, Any]:
  """
  Enforce Stage 0 KB_MEM rule:
  - one old fact key has at most one canonical target_new
  - if key already exists, reuse existing rewrite (memory-first)
  """
  key = (subject, relation, target_old)
  if key not in kb_mem:
    kb_mem[key] = {
      "target_new": target_new,
      "target_label_new": target_label_new,
      "target_aliases_new": target_aliases_new,
      "source_row_index": source_row_index,
      "edit_scope": edit_scope,
    }

  canonical_entry = kb_mem[key]
  kb_mem_sr[(subject, relation)].add(canonical_entry["target_new"])
  return canonical_entry

def _resolve_target_from_hop_index_row(
  df: pd.DataFrame,
  row_id: int,
  subject: Any,
  relation: Any,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
  """Given a row id and queried (subject, relation), resolve matched hop target."""
  row = df.loc[row_id]

  if row["e1"] == subject and row["r1"] == relation:
    return row["e2"], row["e2_label"], row["e2_aliases"]
  if row["e2"] == subject and row["r2"] == relation:
    return row["e3"], row["e3_label"], row["e3_aliases"]

  return None, None, None

def resolve_hop2_target(
  df: pd.DataFrame,
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  hop_index: Dict[Tuple[Any, Any], list[int]],
  entity_lookup: Dict[Any, Dict[str, Any]],
  subject: Any,
  relation: Any,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
  """
  Resolve hop-2 target by Stage 0 rule:
  - memory-first via KB_MEM[(subject, relation)]
  - dataset fallback via smallest row_id in HOP_INDEX[(subject, relation)]
  
  expectation: partner i and partner j hop-2 relation is the same 
  """
  sr_key = (subject, relation)
  
  if sr_key in kb_mem_sr:
    target_new = sorted(kb_mem_sr[sr_key])[0]
    target_label_new = entity_lookup.get(target_new, {}).get("label", pd.NA)
    target_aliases_new = entity_lookup.get(target_new, {}).get("aliases", pd.NA)
    return target_new, target_label_new, target_aliases_new
  
  candidate_row_ids = hop_index.get(sr_key, [])
  if not candidate_row_ids:
    return None, None, None
  
  row_id = min(candidate_row_ids)
  return _resolve_target_from_hop_index_row(df, row_id, subject, relation)

def resolve_effective_subject(
  df: pd.DataFrame,
  row_id: int,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  entity_lookup: Dict[Any, Dict[str, Any]],
) -> Any:
  """
  Stage 0 helper for e3 flow:
  - if (e1, r1, e2) has memory rewrite, set e2_new and return e2_eff=e2_new
  - else return original e2
  """
  row = df.loc[row_id]
  key = (row["e1"], row["r1"], row["e2"])

  if key in kb_mem:
    entry = kb_mem[key]
    _set_new_target(
      df,
      row_id,
      "e2",
      entry["target_new"],
      entity_lookup,
      entry.get("target_label_new"),
      entry.get("target_aliases_new"),
    )
    return entry["target_new"]

  return row["e2"]

def evaluate_row_for_e3(
  df: pd.DataFrame,
  row_id: int,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  protected_facts: Set[Tuple[Any, Any, Any]],
  entity_lookup: Dict[Any, Dict[str, Any]],
) -> Tuple[Literal["blocked", "propagated", "eligible_for_direct_edit"], Any]:
  """Stage 0 reusable e3 evaluator (for both i and j rows)."""
  row = df.loc[row_id]
  if pd.notna(row["swapped_with"]):
    return "blocked", row["e2"]

  e2_eff = resolve_effective_subject(df, row_id, kb_mem, entity_lookup)
  row = df.loc[row_id]
  
  row_role = row["row_role"]

  protected_key = (e2_eff, row["r2"], row["e3"])
  if protected_key in protected_facts:
    if row_role == "edit":
      logging.debug(f"Warning: edit row {row_id} has protected fact conflict but row_role is already edit")

    df.at[row_id, "row_role"] = "propagated_eval"
    return "propagated", e2_eff

  if protected_key in kb_mem:
    if row_role == "edit":
      logging.debug(f"Warning: edit row {row_id} has hop-2 conflict but row_role is already edit")
    
    entry = kb_mem[protected_key]
    _set_new_target(
      df,
      row_id,
      "e3",
      entry["target_new"],
      entity_lookup,
      entry.get("target_label_new"),
      entry.get("target_aliases_new"),
    )
    df.at[row_id, "row_role"] = "propagated_eval"
    return "propagated", e2_eff
  
  if e2_eff != row["e2"]:
    return "blocked", e2_eff

  return "eligible_for_direct_edit", e2_eff

def evaluate_row_for_e2(
  df: pd.DataFrame,
  row_id: int,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  hop_index: Dict[Tuple[Any, Any], list[int]],
  entity_lookup: Dict[Any, Dict[str, Any]],
) -> Tuple[Literal["blocked", "propagated", "eligible_for_direct_edit"], Any]:
  """Stage 0 reusable e2 evaluator (for both i and j rows)."""
  row = df.loc[row_id]
  if pd.notna(row["swapped_with"]):
    return "blocked", row["e2"]
  
  row_role = row["row_role"]

  hop1_key = (row["e1"], row["r1"], row["e2"])
  if hop1_key in kb_mem:
    if row_role == "edit":
      logging.debug(f"Warning: edit row {row_id} has row_role edit, should not be edited again")
    
    entry = kb_mem[hop1_key]
    _set_new_target(
      df,
      row_id,
      "e2",
      entry["target_new"],
      entity_lookup,
      entry.get("target_label_new"),
      entry.get("target_aliases_new"),
    )

    e2_eff = entry["target_new"]
    e3_new, e3_label_new, e3_aliases_new = resolve_hop2_target(
      df,
      kb_mem_sr,
      hop_index,
      entity_lookup,
      e2_eff,
      row["r2"],
    )
    if e3_new is None:
      df.at[row_id, "row_role"] = "unused"
      return "blocked", e2_eff

    _set_new_target(
      df,
      row_id,
      "e3",
      e3_new,
      entity_lookup,
      e3_label_new,
      e3_aliases_new,
    )

    df.at[row_id, "row_role"] = "propagated_eval"

    return "propagated", e2_eff

  return "eligible_for_direct_edit", row["e2"]

def apply_stage2a_e2(
  df: pd.DataFrame,
  idx_group: Dict[Tuple[Any, Any], list[int]],
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  hop_index: Dict[Tuple[Any, Any], list[int]],
  entity_lookup: Dict[Any, Dict[str, Any]],
) -> Dict[str, int]:
  """
  Stage 2A: apply counterfactuals for swapping_strategy == 'e2'.
  """
  stats = {
    "processed": 0,
    "blocked": 0,
    "propagated": 0,
    "edited_pairs": 0,
    "edited_rows": 0,
    "no_partner": 0,
    "pair_hop2_unresolved": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_i in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 2A (e2): apply swaps"):
    stats["processed"] += 1

    state_i, _ = evaluate_row_for_e2(df, row_i, kb_mem, kb_mem_sr, hop_index, entity_lookup)
    if state_i == "blocked":
      stats["blocked"] += 1
      continue
    if state_i == "propagated":
      stats["propagated"] += 1
      continue

    row_i_data = df.loc[row_i]

    selected_j: Optional[int] = None
    selected_row_j_data = None
    selected_e2_new_i = None
    selected_e2_new_j = None
    selected_e3_i: Tuple[Optional[Any], Optional[Any], Optional[Any]] = (None, None, None)
    selected_e3_j: Tuple[Optional[Any], Optional[Any], Optional[Any]] = (None, None, None)

    for row_j in _iter_group_candidates_deterministic(df, row_i, idx_group):
      if row_j == row_i:
        continue
      if pd.notna(df.at[row_j, "swapped_with"]):
        continue
      if not is_valid_swappable_pair(df, row_i, row_j):
        continue

      state_j, _ = evaluate_row_for_e2(df, row_j, kb_mem, kb_mem_sr, hop_index, entity_lookup)
      if state_j != "eligible_for_direct_edit":
        continue

      row_j_data = df.loc[row_j]
      e2_new_i = row_j_data["e2"]
      e2_new_j = row_i_data["e2"]

      if row_i_data["r2"] != row_j_data["r2"]:
        logging.debug(f"Row {row_i} and Row {row_j} should not have different r2 values for e2 swap")
      if ((e2_new_i, row_i_data["r2"], row_i_data["e3"]) in kb_mem) or ((e2_new_j, row_j_data["r2"], row_j_data["e3"]) in kb_mem):
        stats["blocked"] += 1
        continue
      # e3_candidate_i = e2_new_i, row_i_data["r2"], row_j_data["e3"]
      # e3_candidate_j = e2_new_j, row_j_data["r2"], row_i_data["e3"]

      e3_candidate_i = row_j_data["e3"], row_j_data["e3_label"], row_j_data["e3_aliases"]
      e3_candidate_j = row_i_data["e3"], row_i_data["e3_label"], row_i_data["e3_aliases"]


      selected_j = row_j
      selected_row_j_data = row_j_data
      selected_e2_new_i = e2_new_i
      selected_e2_new_j = e2_new_j
      selected_e3_i = e3_candidate_i
      selected_e3_j = e3_candidate_j
      break

    if selected_j is None:
      stats["no_partner"] += 1
      continue

    row_j_data = selected_row_j_data
    e2_new_i = selected_e2_new_i
    e2_new_j = selected_e2_new_j

    _set_new_target(
      df,
      row_i,
      "e2",
      e2_new_i,
      entity_lookup,
      row_j_data["e2_label"],
      row_j_data["e2_aliases"],
    )
    _set_new_target(
      df,
      selected_j,
      "e2",
      e2_new_j,
      entity_lookup,
      row_i_data["e2_label"],
      row_i_data["e2_aliases"],
    )

    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      row_i_data["e1"],
      row_i_data["r1"],
      row_i_data["e2"],
      e2_new_i,
      row_j_data["e2_label"],
      row_j_data["e2_aliases"],
      row_i,
      "hop1",
    )
    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      row_j_data["e1"],
      row_j_data["r1"],
      row_j_data["e2"],
      e2_new_j,
      row_i_data["e2_label"],
      row_i_data["e2_aliases"],
      selected_j,
      "hop1",
    )

    df.at[row_i, "row_role"] = "edit"
    df.at[selected_j, "row_role"] = "edit"
    df.at[row_i, "swapped_with"] = selected_j
    df.at[selected_j, "swapped_with"] = row_i

    e3_new_i, e3_label_new_i, e3_aliases_new_i = selected_e3_i
    _set_new_target(
      df,
      row_i,
      "e3",
      e3_new_i,
      entity_lookup,
      e3_label_new_i,
      e3_aliases_new_i,
    )

    e3_new_j, e3_label_new_j, e3_aliases_new_j = selected_e3_j
    _set_new_target(
      df,
      selected_j,
      "e3",
      e3_new_j,
      entity_lookup,
      e3_label_new_j,
      e3_aliases_new_j,
    )

    stats["edited_pairs"] += 1
    stats["edited_rows"] += 2

  return stats

def apply_stage2b_e3(
  df: pd.DataFrame,
  idx_group: Dict[Tuple[Any, Any], list[int]],
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  protected_facts: Set[Tuple[Any, Any, Any]],
  entity_lookup: Dict[Any, Dict[str, Any]],
) -> Dict[str, int]:
  """
  Stage 2B: apply counterfactuals for swapping_strategy == 'e3'.
  """
  stats = {
    "processed": 0,
    "blocked": 0,
    "propagated": 0,
    "edited_pairs": 0,
    "edited_rows": 0,
    "no_partner": 0,
    "same_e3_skips": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_i in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 2B (e3): apply swaps"):
    stats["processed"] += 1

    state_i, e2_eff_i = evaluate_row_for_e3(df, row_i, kb_mem, protected_facts, entity_lookup)
    if state_i == "blocked":
      stats["blocked"] += 1
      continue
    if state_i == "propagated":
      stats["propagated"] += 1
      continue

    row_i_data = df.loc[row_i]
    selected_j: Optional[int] = None
    selected_e2_eff_j: Optional[Any] = None

    for row_j in _iter_group_candidates_deterministic(df, row_i, idx_group):
      if row_j == row_i:
        continue
      if pd.notna(df.at[row_j, "swapped_with"]):
        continue
      if not is_valid_swappable_pair(df, row_i, row_j):
        continue

      row_j_data = df.loc[row_j]
      if row_j_data["e3"] == row_i_data["e3"]:
        stats["same_e3_skips"] += 1
        continue

      state_j, e2_eff_j = evaluate_row_for_e3(df, row_j, kb_mem, protected_facts, entity_lookup)
      if state_j != "eligible_for_direct_edit":
        continue

      selected_j = row_j
      selected_e2_eff_j = e2_eff_j
      break

    if selected_j is None:
      stats["no_partner"] += 1
      continue

    row_j_data = df.loc[selected_j]

    e3_new_i = row_j_data["e3"]
    e3_new_j = row_i_data["e3"]

    _set_new_target(
      df,
      row_i,
      "e3",
      e3_new_i,
      entity_lookup,
      row_j_data["e3_label"],
      row_j_data["e3_aliases"],
    )
    _set_new_target(
      df,
      selected_j,
      "e3",
      e3_new_j,
      entity_lookup,
      row_i_data["e3_label"],
      row_i_data["e3_aliases"],
    )

    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      e2_eff_i,
      row_i_data["r2"],
      row_i_data["e3"],
      e3_new_i,
      row_j_data["e3_label"],
      row_j_data["e3_aliases"],
      row_i,
      "hop2",
    )
    register_kb_rewrite(
      kb_mem,
      kb_mem_sr,
      selected_e2_eff_j,
      row_j_data["r2"],
      row_j_data["e3"],
      e3_new_j,
      row_i_data["e3_label"],
      row_i_data["e3_aliases"],
      selected_j,
      "hop2",
    )

    df.at[row_i, "row_role"] = "edit"
    df.at[selected_j, "row_role"] = "edit"
    df.at[row_i, "swapped_with"] = selected_j
    df.at[selected_j, "swapped_with"] = row_i

    protected_facts.add((row_i_data["e1"], row_i_data["r1"], e2_eff_i))
    protected_facts.add((row_j_data["e1"], row_j_data["r1"], selected_e2_eff_j))

    stats["edited_pairs"] += 1
    stats["edited_rows"] += 2

  return stats

def apply_stage3_conflict_resolution(
  df: pd.DataFrame,
  kb_mem: Dict[Tuple[Any, Any, Any], Dict[str, Any]],
  kb_mem_sr: Dict[Tuple[Any, Any], Set[Any]],
  hop_index: Dict[Tuple[Any, Any], list[int]],
  entity_lookup: Dict[Any, Dict[str, Any]],
  swapping_strategy: Literal["e2", "e3"],
) -> Dict[str, int]:
  """
  Stage 3: strategy-agnostic conflict resolution from memory/index.
  """
  stats = {
    "processed": 0,
    "hop1_propagated": 0,
    "hop2_from_memory_sr": 0,
    "hop2_from_hop_index": 0,
    "hop2_from_exact_memory": 0,
    "set_unused": 0,
    "edit_row_hop1_conflict": 0,
  }

  sorted_row_ids = sorted(df.index.tolist())
  for row_id in tqdm(sorted_row_ids, total=len(sorted_row_ids), desc="Stage 3: resolve conflicts"):
    stats["processed"] += 1
    set_unused = False

    row = df.loc[row_id]
    e1 = row["e1"]
    r1 = row["r1"]
    e2 = row["e2"]
    r2 = row["r2"]
    e3 = row["e3"]
    row_role = row["row_role"]

    hop1_key = (e1, r1, e2)
    if hop1_key in kb_mem:
      entry = kb_mem[hop1_key]
      target_new = entry["target_new"]
      
      if swapping_strategy == "e2" and row_role == "propagated_eval" and row["e2_new"] != target_new:
        logging.debug(f"Warning: row={row_id} with row_role propagated_eval should not have hop-1 conflict")
      if row_role == "edit" and row["e2_new"] != target_new:
        logging.debug(f"Warning: row={row_id} with row_role edit shouldn't have hop-1 conflict")
      
      if pd.isna(row["e2_new"]) or row["e2_new"] != target_new:
        _set_new_target(
          df,
          row_id,
          "e2",
          target_new,
          entity_lookup,
          entry.get("target_label_new"),
          entry.get("target_aliases_new"),
        )
        if row_role == "unused":
          df.at[row_id, "row_role"] = "propagated_eval"
        stats["hop1_propagated"] += 1

    e2_eff = df.at[row_id, "e2_new"] if pd.notna(df.at[row_id, "e2_new"]) else e2

    if e2_eff != e2:
      if swapping_strategy == "e3" and row_role == "edit":
        logging.debug(f"Warning: edit row {row_id} with row_role edit has hop-2 conflict from hop index")
      sr_key = (e2_eff, r2)

      if sr_key in kb_mem_sr and kb_mem_sr[sr_key]:
        e3_new = sorted(kb_mem_sr[sr_key])[0]
        e3_label_new = entity_lookup.get(e3_new, {}).get("label", pd.NA)
        e3_aliases_new = entity_lookup.get(e3_new, {}).get("aliases", pd.NA)
        if pd.isna(row["e3_new"]) or row["e3_new"] != e3_new:
          _set_new_target(
            df,
            row_id,
            "e3",
            e3_new,
            entity_lookup,
            e3_label_new,
            e3_aliases_new,
          )
          stats["hop2_from_memory_sr"] += 1
      else:
        candidate_row_ids = hop_index.get(sr_key, [])
        if candidate_row_ids:
          row_ref = min(candidate_row_ids)
          e3_new, e3_label_new, e3_aliases_new = _resolve_target_from_hop_index_row(
            df,
            row_ref,
            e2_eff,
            r2,
          )
          if e3_new is not None:
            if pd.isna(row["e3_new"]) or row["e3_new"] != e3_new:
              _set_new_target(
                df,
                row_id,
                "e3",
                e3_new,
                entity_lookup,
                e3_label_new,
                e3_aliases_new,
              )
              if row_role == "unused":
                df.at[row_id, "row_role"] = "propagated_eval"
              stats["hop2_from_hop_index"] += 1
          elif row_role == "propagated_eval":
            set_unused = True
        elif row_role == "propagated_eval":
          set_unused = True

    else:
      hop2_exact_key = (e2_eff, r2, e3)
      if hop2_exact_key in kb_mem:
        entry = kb_mem[hop2_exact_key]

        if pd.isna(row["e3_new"]) or row["e3_new"] != entry["target_new"]:
          if swapping_strategy == "e3" and row_role == "edit":
            logging.debug(f"Warning: edit row {row_id} with row_role edit should not have hop-2 conflict")
          _set_new_target(
            df,
            row_id,
            "e3",
            entry["target_new"],
            entity_lookup,
            entry.get("target_label_new"),
            entry.get("target_aliases_new"),
          )
          if row_role == "unused":
            df.at[row_id, "row_role"] = "propagated_eval"
          stats["hop2_from_exact_memory"] += 1

    if set_unused:
      df.at[row_id, "row_role"] = "unused"
      stats["set_unused"] += 1

  return stats

def apply_stage4_export(df: pd.DataFrame, output_path: str | Path) -> Dict[str, Any]:
  """
  Stage 4: final benchmark export.
  - write only the full trace table to output_path
  - report row_role statistics
  """
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  full_df = df.sort_index()
  edit_df = full_df[full_df["row_role"] == "edit"]
  propagated_df = full_df[full_df["row_role"] == "propagated_eval"]
  active_df = full_df[full_df["row_role"] != "unused"]

  full_df.to_csv(output_path, index=False)

  return {
    "full_path": str(output_path),
    "full_rows": int(len(full_df)),
    "edit_rows": int(len(edit_df)),
    "propagated_rows": int(len(propagated_df)),
    "active_rows": int(len(active_df)),
    "unused_rows": int((full_df["row_role"] == "unused").sum()),
    "df_new": full_df,
  }

# Stage 0: initialize runtime state
df = initialize_stage0_columns(df)

KB_MEM: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
KB_MEM_SR: Dict[Tuple[Any, Any], Set[Any]] = defaultdict(set)
PROTECTED_FACTS: Set[Tuple[Any, Any, Any]] = set()

ENTITY_LOOKUP = build_entity_lookup(df)
HOP_INDEX = build_hop_index(df)

print(
  f"Stage 0 complete | rows={len(df)} | hop_keys={len(HOP_INDEX)} | entity_ids={len(ENTITY_LOOKUP)}"
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
    HOP_INDEX,
    ENTITY_LOOKUP,
  )
  print(
    "Stage 2A complete | "
    f"processed={stage2a_stats['processed']} | "
    f"blocked={stage2a_stats['blocked']} | "
    f"propagated={stage2a_stats['propagated']} | "
    f"edited_pairs={stage2a_stats['edited_pairs']} | "
    f"edited_rows={stage2a_stats['edited_rows']} | "
    f"no_partner={stage2a_stats['no_partner']} | "
    f"pair_hop2_unresolved={stage2a_stats['pair_hop2_unresolved']}"
  )
else:
  stage2b_stats = apply_stage2b_e3(
    df,
    IDX_GROUP,
    KB_MEM,
    KB_MEM_SR,
    PROTECTED_FACTS,
    ENTITY_LOOKUP,
  )
  print(
    "Stage 2B complete | "
    f"processed={stage2b_stats['processed']} | "
    f"blocked={stage2b_stats['blocked']} | "
    f"propagated={stage2b_stats['propagated']} | "
    f"edited_pairs={stage2b_stats['edited_pairs']} | "
    f"edited_rows={stage2b_stats['edited_rows']} | "
    f"no_partner={stage2b_stats['no_partner']} | "
    f"same_e3_skips={stage2b_stats['same_e3_skips']}"
  )

stage3_stats = apply_stage3_conflict_resolution(
  df,
  KB_MEM,
  KB_MEM_SR,
  HOP_INDEX,
  ENTITY_LOOKUP,
  args.swapping_strategy,
)
print(
  "Stage 3 complete | "
  f"processed={stage3_stats['processed']} | "
  f"hop1_propagated={stage3_stats['hop1_propagated']} | "
  f"hop2_from_memory_sr={stage3_stats['hop2_from_memory_sr']} | "
  f"hop2_from_hop_index={stage3_stats['hop2_from_hop_index']} | "
  f"hop2_from_exact_memory={stage3_stats['hop2_from_exact_memory']} | "
  f"set_unused={stage3_stats['set_unused']} | "
  f"edit_row_hop1_conflict={stage3_stats['edit_row_hop1_conflict']}"
)

# %%

stage4_stats = apply_stage4_export(df, args.output_path)
print(
  "Stage 4 complete | "
  f"full_rows={stage4_stats['full_rows']} | "
  f"edit_rows={stage4_stats['edit_rows']} | "
  f"propagated_rows={stage4_stats['propagated_rows']} | "
  f"active_rows={stage4_stats['active_rows']} | "
  f"unused_rows={stage4_stats['unused_rows']} | "
  f"full_path={stage4_stats['full_path']}"
)

# %%
result = stage4_stats["df_new"]

# %%
# ================================
# 📊 ANALYSIS + VALIDATION SUITE
# ================================

def analysis_summary(df, result):
    print("="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)

    print(f"Original rows: {len(df)}")
    print(f"Result rows:   {len(result)}")
    print()

    print("---- Counterfactual Coverage ----")
    print(f"Rows with e2_new: {result['e2_new'].notna().sum()}")
    print(f"Rows without e2_new: {result['e2_new'].isna().sum()}")

    print(f"Rows with e3_new: {result['e3_new'].notna().sum()}")
    print(f"Rows without e3_new: {result['e3_new'].isna().sum()}")
    print()

    print("---- Row Type Breakdown ----")
    print(result["row_role"].value_counts(dropna=False))
    print()

    print("---- Uniqueness ----")
    print(f"Unique e2: {result['e2'].nunique()}")
    print(f"Unique e2_new: {result['e2_new'].nunique()}")

    print(f"Unique e3: {result['e3'].nunique()}")
    print(f"Unique e3_new: {result['e3_new'].nunique()}")
    print()

    print("="*60)
analysis_summary(df, result)


# %%
# check overlap
mask = result["e2_new"].notna()
inconsistent = result.loc[mask & (result["e2_new"] == result["e2"])]
print("e2_new overlap:", len(inconsistent))

mask = result["e3_new"].notna()
inconsistent = result.loc[mask & (result["e3_new"] == result["e3"])]
print("e3_new overlap:", len(inconsistent))

inconsistent[["source_prompt","e1_label", "e2_label", "e3_label","e2_label_new" ,"e3_label_new", "row_role"]].head()

# %%
def test_knowledge_consistency():
  """check if there are multiple e2_new values for the same (e1, r1, e2) group"""
  group_check = result.groupby(["e1","r1","e2"])["e2_new"].nunique()
  assert (group_check > 1).sum() == 0

test_knowledge_consistency()

# %%
# %%
def find_invalid_r2_rows(df, result):
    """
    Return rows where (e2_new, r2, e3_new) is NOT in original dataset
    """

    valid_knowledge = set(zip(df["e2"], df["r2"], df["e3"]))

    mask = result["e2_new"].notna() & result["e3_new"].notna()

    invalid_mask = result.loc[mask].apply(
        lambda r: (r["e2_new"], r["r2"], r["e3_new"]) not in valid_knowledge,
        axis=1
    )

    invalid_rows = result.loc[mask].loc[invalid_mask]

    print(f"Invalid rows (violating knowledge): {len(invalid_rows)}")

    return invalid_rows



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

invalid_rows = find_invalid_r2_rows(df, result)
invalid_rows[
  ["source_prompt","e1_label", "e2_label", "e3_label","e2_label_new" ,"e3_label_new", "row_role", "swapped_with"]
]

# %%
original_count = len(df)
e2_new_count = result["e2_new"].notna().sum()
e3_new_count = result["e3_new"].notna().sum()

print(f"len(df): {original_count}")
# for some rows no B exists that match condition so such rows are dropped
# That is why len(result) is smaller than len(df)
print(f"len(result): {len(result)}")

print(f"Counterfactuals with e2_new generated: {e2_new_count}")
print(f"Number of rows wihtout e2_new: {original_count - e2_new_count}")

print(f"Counterfactuals with e3_new generated: {e3_new_count}")
print(f"Number of rows without e3_new: {original_count - e3_new_count}")

# %%
# %%
def count_consecutive_edit_rows(result):
    """
    Count how many edit rows are immediately followed by another edit row.
    """

    # reset index to ensure sequential access
    df_seq = result.reset_index(drop=True)

    count = 0
    indices = []

    for i in range(len(df_seq) - 1):
        if df_seq.loc[i, "row_role"] == "edit" and df_seq.loc[i+1, "row_role"] == "edit":
            count += 1
            indices.append(i)

    return count


# run
print(f" Consecutive edit rows: {count_consecutive_edit_rows(result)}")

# %%
def no_invalid_edit_rows():
    """check if there are any invalid rows with row_type == edit"""
    assert ((result["row_role"] == "edit") & (result["e2_new"].isna())).sum() == 0
    missing_e3_edit = result[
        (result["row_role"] == "edit") & (result["e3_new"].isna())
    ]
    print(f"{len(missing_e3_edit)} edit rows with missing e3_new values")

no_invalid_edit_rows()