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
         - return `state = blocked`
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
    - set `e2_new_i <- e2_j` and `e2_new_j <- e2_i`
    - write both hop-1 rewrites to `KB_MEM`:
     - `KB_MEM[(e1_i, r1_i, e2_i)] = e2_new_i`
     - `KB_MEM[(e1_j, r1_j, e2_j)] = e2_new_j`
    - set `row_role_i = "edit"`, `row_role_j = "edit"`
    - set `swapped_with_i = j` and `swapped_with_j = i`
    - recompute hop-2 for both sides:
     - `e3_new_i` from `KB_MEM_SR[(e2_new_i, r2_i)]` else smallest row_id in `HOP_INDEX[(e2_new_i, r2_i)]`
     - `e3_new_j` from `KB_MEM_SR[(e2_new_j, r2_j)]` else smallest row_id in `HOP_INDEX[(e2_new_j, r2_j)]`

Stage 2B: Apply counterfactuals when swapping_strategy == "e3"
- Process rows in ascending index i:
  1) run `evaluate_row_for_e3(row_i)`:
    - if state is `blocked`, skip row i
    - if state is `propagated`, set row_role_i="propagated_eval" and continue to next i
  2) find partner candidates from `HOP_INDEX[(e2_eff_i, r2_i)]`, excluding invalid/self and requiring `e3_j != e3_i`
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
    - only when `target_new != e2`:
      - set `e2_new = target_new`
      - if `row_role` is `unused`, set `row_role = propagated_eval`
  2) Define effective subject:
    - `e2_eff = e2_new` if present else `e2`
  3) Choose exactly one hop-2 branch:
    - Branch A: (`row_role == "edit"` for either `swapping_strategy`)
      - if `e2_eff != e2` on this edit row, print warning `edit_row_hop1_conflict`
      - stop processing this
    - Branch B: (all other rows)
      - if `e2_eff != e2`:
        - if `(e2_eff, r2)` in `KB_MEM_SR`, set `e3_new` from memory
        - else if `HOP_INDEX[(e2_eff, r2)]` has at least one row_id:
          - set `e3_new` from smallest row_id in `HOP_INDEX[(e2_eff, r2)]`
        - else:
          - if `row_role == propagated_eval`, set `set_unused = True`
      - else if `(e2_eff, r2, e3)` exists in `KB_MEM`:
        - set `e3_new` from memory
        - if `row_role == unused`, set `row_role = propagated_eval`
  4) Branch B outcome rule:
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