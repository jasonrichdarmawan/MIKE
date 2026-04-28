# %%
# Import necessary libraries and modules

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from IPython.display import display

from analysis.create_dataset import RELATION_AFTER_ENTITY_TEMPLATES

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
  args, _ = parser.parse_known_args()
  return args


args = parse_args()

# %%
# Load the two-hop dataset

df = pd.read_csv(args.input_path)

# %%
# inspect the dataset structure and columns to understand how to generate coutnerfactuals

"""
columns:
- `e1`: the entity id in the first hop
  - 'Q206'
    - https://www.wikidata.org/wiki/Q206 
- `r1`: the relation id in the first hop
  - 'P27'
    - https://www.wikidata.org/wiki/Property:P27
- `e2`: the entity id in the second hop
   - 'Q16'
- `r2`: the relation id in the second hop
   - 'P35'
- `e3`: the entity id in the third hop
   - 'Q43274'
- `e1_label`: the label of `e1`
  - 'Stephen Harper'
- `e2_label`: the label of `e2`
  - 'Canada'
- `e3_label`: the label of `e3`
  - 'Charles III of the United Kingdom'
- `e1_aliases`: the aliases of `e1`
  - ['HE Stephen Harper', 'Stephen Joseph Harper', 'The Right Hon. Stephen Harper', 'H.E. Stephen Harper']
- `e2_aliases`: the aliases of `e2`
  - ['CA', 'CAN', 'Dominion of Canada', 'Canada (country)', 'Commonwealth of Canada', 'ISO 3166-1:CA', 'ᑲᓇᑕ']
- `e3_aliases`: the aliases of `e3`
  - ['C III', 'Charles Mountbatten-Windsor', 'Charles Philip Arthur George Mountbatten-Windsor', 'Charles Philip Arthur George Windsor', 'Charles Philip Arthur George, Prince of Wales', 'Charles the Third', 'Charles Wales', 'His Majesty Charles the Third, by the Grace of God, of the United Kingdom of Great Britain and Northern Ireland and of His other Realms and Territories King, Head of the Commonwealth, Defender of the Faith', 'His Majesty The King', 'His Royal Highness Charles III of Great Britain', 'HM King Charles III', 'King Charles', 'Prince Charles of Edinburgh', 'Prince Charles Philip Arthur George of Edinburgh', 'Prince Charles Philip Arthur George, Prince of Wales', 'Prince Charles, Duke of Rothesay', 'Prince Charles, Prince of Wales', 'The Prince Charles Philip Arthur George, Prince of Wales', 'The Prince Charles, Prince of Wales', 'Charles III of England', 'Chuck', 'King Charles III', 'Prince Charles', 'King of England', 'His Majesty', 'Charles Windsor']
- `e1_type`: the type of `e1`
  - 'human'
- `r1_type`: the type of `r1`
  - 'country of citizenship'
- `e2_type`: the type of `e2`
  - 'country'
- `r2_type`: the type of `r2`
  - 'head of state'
- `e3_type`: the type of `e3`
  - 'human'
- `e1_popularity`: the average Wikipedia pageviews for the entity `e_1` (mean daily pageviews over 7 days)
  - 1025.0
- `e2_popularity`: the average Wikipedia pageviews for the entity `e_2` (mean daily pageviews over 7 days)
  - 179.0
- `e3_popularity`: the average Wikipedia pageviews for the entity `e_3` (mean daily pageviews over 7 days)
  - 132.0
- `r1_template`: the template of the relation `r1` (the relation with entity placeholders)
  - Prompt relation-before-entity:
    - 'the country of citizenship of {}'
- `r2_template`: the template of the relation `r2` (the relation with entity placeholders)
  - Prompt relation-before-entity:
    - 'the head of state of {}'
  - Prompt relation-after-entity:
    - '{} is the head of state of'
- `source_prompt`: the multi-hop prompt to be evaluated.
  - Prompt relation-before-entity format: [r2_template, r1_template, e1_label] + ' is'
    - 'the head of state of the country of citizenship of Stephen Harper is'
  - Prompt format: [r2_template, r1_template, e1_label]
    - 'the country of citizenship of Stephen Harper is the head of state of'
      - Note: The source code to create the `Two Hop` dataset indicates that this format exists, but it is not present in the dataset. The unnatural sentence (for some combination of r2_template and r1_template) might explain why it is not present in the dataset. We will generate counterfactuals based on the relation-before-entity format.
      - A natural sentence of relation-after-entity format might be the following:
        - Suppose `r1_template` is 'the mother of {}', `r2_template` is '{} has a spouse named'. Then, the source prompt would be 'the mother of Stephen Harper has a spouse named"
"""

display(df.head())

# %%
# check if any instances with relation-after-entity templates exist in the dataset

mask = ~df["r2_template"].astype(str).str.endswith(" is")
print(f"{mask.sum()} rows have r2_template that does not end with ' is'.")

templates = set(RELATION_AFTER_ENTITY_TEMPLATES.values())

r2 = df["r2_template"]
mask = r2.isin(templates)

print(f"{mask.sum()} rows match known 'relation-after-entity' templates; {len(df)-mask.sum()} do not.")
# show top non-matching templates to inspect
display(r2[~mask].value_counts().head(n=20))

# %%
# Get instances where `r1_type` is 'country of citizenship' to find potential counterfactual pairs

mask = df["r1_type"].astype(str) == "country of citizenship"
display(df[mask].head(n=10))

# %%
# Check if `r1_type` and `r2_type` group have multiple values of `e1_type`. This is to ensure that we can swap `e2` values to generate counterfactuals
"""
Findings:
- `source_prompt`:
  - 'the composer of the national anthem of Albert II, Prince of Monaco is'
  - `e1_type` is 'human'
  - Note: Even though it is not a natural sentence, GPT-OSS-120B answer is 'Charles Alphonse' instead of 'Charles Albrecht'.
  - This instance exists because `Albert II, Prince of Monaco` in Wikidata does have direct relation to the `anthem` of 'Hymne monégasque'
- `r1_type` and `r2_type` -> `e1_type` is many-to-many mapping
"""

display(df.groupby(["r1_type", "r2_type"])["e1_type"].unique())

mask = (df["r1_type"] == "anthem") & (df["r2_type"] == "composer")
display(df[mask][["e1", "e1_type", "e2_type", "e2_label", "e3_label", "source_prompt"]].head(n=10))

# %%
# Check if `r1_type` have multiple values of `e2_type`. This is to ensure that we can swap `e2` values to generate counterfactuals
"""
Findings:
- `source_prompt`:
  - 'the composer of the national anthem of Fukuoka is'
  - `e1_type` is 'city'
  - Note: GPT-OSS-120B can't answer Fukuoka's anthem.
  - This instance exists because the `r1_template` is hardcoded to 'the national anthem of {}' for all `r1_type` of 'anthem', even though some `r1_type` of 'anthem' do not have `e2_type` of 'country'
- `r1_type` -> `e2_type` is many-to-many mapping
- Verdict: Considering the existing `source_prompt` (e.g., 'the founder of the author of Google Patents is', which is odd to use the word 'author', but some model still can answer it e.g. GPT-OSS-120B), we can swap `e2` values within the same `r1_type` and `r2_type`, regardless of `e2_type` because `e2_type` is generic, the problem lies with the `r1_template` and `r2_template`.
"""
display(df.groupby("r1_type")["e2_type"].unique())

mask = df["r1_type"] == "anthem"
display(df[mask][["e1", "e1_type", "e2_type", "e2_label", "e3_label", "source_prompt"]].head(n=10))

mask = df["r1_type"] == "author"
display(df[mask][["e1", "e1_type", "e2_type", "e2_label", "e3_label", "source_prompt"]].groupby("e2_type").head(1).head(n=10))

# %%
# Check if each group of `r1_type` and `r2_type` have enough instances to swap `e2` values to generate counterfactuals
"""
Findings:
- row indices 12738 and 12739 have the same `source_prompt` but different `e3_label` (there are more indices)
- There are 30 groups of `r1_type` and `r2_type` that have less than 2 instances, and 44 groups that have less than 2 distinct `e2` values
  - Example:
    - row indices 3337 and 3513 have different `e1`, but same `e2` and `e3`. This means that we can't swap `e2` values to generate counterfactuals for these two rows
    - row indices 4305 and 4307 have different `e1`, same `e2`, but different `e3`. At a glance, it seems that we can swap `e3` values to generate counterfactuals, but we need to make sure `e1` != `e3`
      - row index 4305:
        - `e1_label`: 'Ya Lil'
        - `e3_label`: 'Ali Farah'
      - row index 4307:
        - `e1_label`: 'Ali Farah'
        - `e3_label`: 'Ya Lil'
- Verdict: Although the example above is non-sensical, there is a chance that `e1` and `e3` have bi-directional relation. Swapping between `e3` might need to check whether `e3` != `e1`
"""

groups = df.groupby(["r1_type", "r2_type"])["e2"] # tells panda "when you aggregate/transform/apply, do it on `e2`"
display(groups.size()[groups.size() < 2]) # count of rows for each group of `r1_type` and `r2_type`
display(groups.nunique()[groups.nunique() < 2]) # distinct `e2` count for each group of `r1_type` and `r2_type`. This includes singleton groups but also groups with many rows where every row has the same `e2`

mask = (groups.transform("size") < 2)
display(df[mask][["e1_label", "r1_type", "r2_type", "e2_label", "source_prompt"]])

mask = (groups.transform("nunique") < 2)
display(df[mask][["e1", "e1_label", "r1_type", "r2_type", "e2_type", "e2_label", "e3", "e3_label", "source_prompt"]])

def has_no_valid_partner(group: pd.DataFrame):
  e1 = group["e1"].to_numpy()
  e2 = group["e2"].to_numpy()
  partner = (e1[:, None] != e1[None, :]) & (e2[:, None] != e2[None, :])
  return pd.Series(~partner.any(axis=1), index=group.index)

mask_no_partner = df.groupby(["r1_type", "r2_type"], group_keys=False).apply(has_no_valid_partner)
unswappable_instances = df[mask_no_partner]
display(unswappable_instances[["r1_type", "r2_type", "e1_label", "e2_label", "e3_label", "source_prompt"]])

# %%
# check if a instance have different `e1` (so swappable candidate exists)
# but candidate `e1` equals current rows `e2`
"""
Findings:
- instance A has `e1` that might be `e2` in instance B.
"""

def partner_flags(group: pd.DataFrame) -> pd.DataFrame:
  e1 = group["e1"].to_numpy()
  e2 = group["e2"].to_numpy()

  # normal swap candidate
  candidate = (e1[:, None] != e1[None, :]) & (e2[:, None] != e2[None, :])

  # problematic condition: other.e1 == this.e2
  bad_cross = (e2[:, None] == e1[None, :])

  has_bad_partner = (candidate & bad_cross).any(axis=1)
  has_clean_partner = (candidate & ~bad_cross).any(axis=1)

  return pd.DataFrame(
    {
      "has_bad_partner": has_bad_partner,
      "has_clean_partner": has_clean_partner,
      "no_valid_partner": ~candidate.any(axis=1),
    },
    index=group.index,
  )

flags = (
  df.groupby(["r1_type", "r2_type"], group_keys=False)
    .apply(partner_flags)
)

# rows that have at least one swappable candidate, but at least one is "bad" (other.e1 == this.e2)
problem_rows = df[flags["has_bad_partner"]]

# rows that have candidates but all candidates are bad
only_bad_rows = df[flags["has_bad_partner"] & ~flags["has_clean_partner"]]

display(df[flags["no_valid_partner"]][["r1_type", "r2_type", "e1_label", "e2_label", "e3_label", "source_prompt"]])
display(problem_rows[["r1_type", "r2_type", "e1_label", "e2_label", "e3_label", "source_prompt"]])
display(only_bad_rows[["r1_type", "r2_type", "e1_label", "e2_label", "e3_label", "source_prompt"]])

# %%
# check if the groups with less than 2 distinct `e2` values, but have more than 2 distinct `e3` values, which means we can swap `e3` values to generate counterfactuals
"""
Findings:
- Groups with less than 2 distinct `e2` values but more than 2 distinct `e3` values cannot be used to generate counterfactuals by swapping `e3` values
  - This is because there is only 1 pair with the same `r1_type` and `r2_type`, different `e1`, same `e2`, but different `e3`. However, the `e1` of instance 1 and the `e3` of instance 2 are the same, which means we can't swap `e3` values to generate counterfactuals for these two instances.
  - the rest is the same `e1`, `r1_type`, `r2_type`, `e2`, but different `e3`.
  - These instances exists because the answer to [r2 r1 e1] `source_prompt` have multiple `e3` values in Wikidata e.g. (Larry Page and Sergey Brin are both founders of Google Patents), but each instance only includes one of the `e3` values and duplicates the [r2 r1 e1] `source_prompt`.
- Verdict:
  - We can't generate counterfactuals by swapping `e3` values for the remaining groups with less than 2 distinct `e2`, but have more than 2 distinct `e3` values.
  
Question:
1. if we change [subject, relation, target -> target_new],
   - will it cause knowledge conflict with [e1, r1] or [e2, r2] of other instances? Yes, because other instance might have the same [subject, relation] in [e1, r1] or [e2, r2]
   - so, if we change [e1, r1, e2 -> target_new], we need to make sure that the new [subject, relation, target -> target_new] is reflected to other instances with the same [subject, relation] in [e1, r1] or [e2, r2] to avoid knowledge conflict.
     - fixing [e1, r1] of other instances to [e2 -> target_new] requires fixing [e3 -> e3_new] of that instance as well. let's say instance A
       - to do this, we need to find instances with [subject, relation] and use their [target] to replace [e3 -> target_new] of the instance A.
       - there is a chance that there is no [subject, relation] of other instance to be used as reference (note: we should look [e1, r1] and [e2, r2] of other instances), which means we can't fix the knowledge conflict.
       - so, during the construction of counterfactuals, we need to have:
         - a column to indicate whether the instance have been swapped, and so should not be swapped again
         - a memory of [subject, relation, target, target_new] to check when we are about to swap, if [e1, r1] is in the memory bank and the target's instance is not target_new, then resolve the knowledge conflict
         - a column to indicate knowledge conflict can not be resolved, so that we can filter out these instances in the final counterfactual dataset
   - if we change [e2, r2, e3 -> target_new], we only need to change it.
   
Verdict:
- We will generate counterfactuals by swapping `e2` values in group of `r1_type` and `r2_type`, until all groups are swapped.
- Store the counterfactual in a memory bank, so [subject, relation, target, target_new] is only changed once
- After we swap `e2` values in all groups, we will check if there is knowledge conflict in [e1, r1] or [e2, r2] of that instance. If there is knowledge conflict, we will resolve it. If we can't resolve it, we will mark it in the column `knowledge_conflict` as True
"""

def has_valid_partner(group: pd.DataFrame, partner: str):
  e1 = group["e1"].to_numpy()
  e3 = group[partner].to_numpy()
  partner = (e1[:, None] != e1[None, :]) & (e3[:, None] != e3[None, :])
  return pd.Series(partner.any(axis=1), index=group.index)

groups_e3 = unswappable_instances.groupby(["r1_type", "r2_type"])["e3"]
display(groups_e3.nunique()[groups_e3.nunique() >= 2])

mask_e3 = (groups_e3.transform("nunique") >= 2)
mask_e3 = mask_e3[mask_e3 == True]
display(df.loc[mask_e3.index][["e1", "e1_label", "r1_type", "r2_type", "e2_type", "e2_label", "e3", "e3_type", "e3_label", "source_prompt"]])

mask_swappable = unswappable_instances.groupby(["r1_type", "r2_type"], group_keys=False).apply(lambda g: has_valid_partner(g, "e3"))
display(df[mask][mask_swappable][["e1", "e1_label", "r1_type", "r2_type", "e2_type", "e2_label", "e3", "e3_type", "e3_label", "source_prompt"]])

# %%