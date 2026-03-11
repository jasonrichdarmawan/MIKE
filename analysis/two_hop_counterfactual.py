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
    "--output_path",
    type=str,
    default=Path(__file__).resolve().parents[1] / "datasets" / "two_hop_cf.csv",
    help="Path to save the generated counterfactuals CSV dataset.",
  )
  args, _ = parser.parse_known_args()
  return args

args = parse_args()

# %%
# Load the two-hop dataset

df = pd.read_csv(args.input_path)

# %%

def shift_group(g):
    """
    shift group by one position, so that each row gets the next row's values
    so each row will be paired with next row in group
    """
    shifted = g.shift(-1)
    shifted.iloc[-1] = g.iloc[0]
    return shifted

# %%

rows = []

# group dataframe by r1
# grouping ensures e2 and e3 values are swapped only within same r1
r1_groups = df.groupby("r1")

for _, r1_g in r1_groups:
    # grouping again ensures values are swapped only when r1 and r2 both match
    r2_groups = r1_g.groupby("r2")
    for _, r2_g in r2_groups:
        match_group = shift_group(r2_g)
        
        # avoid writing if e2_new == e2 AND e3_new == e3
        mask = (r2_g["e2"] != match_group["e2"]) | (r2_g["e3"] != match_group["e3"])
        
        r2_g["e2_new"] = None
        r2_g["e3_new"] = None
        r2_g["e2_label_new"] = None
        r2_g["e3_label_new"] = None
        r2_g["e2_aliases_new"] = None
        r2_g["e3_aliases_new"] = None
        r2_g.loc[mask, "e2_new"] = match_group.loc[mask, "e2"].values
        r2_g.loc[mask, "e3_new"] = match_group.loc[mask, "e3"].values
        r2_g.loc[mask, "e2_label_new"] = match_group.loc[mask, "e2_label"].values
        r2_g.loc[mask, "e3_label_new"] = match_group.loc[mask, "e3_label"].values
        r2_g.loc[mask, "e2_aliases_new"] = match_group.loc[mask, "e2_aliases"].values
        r2_g.loc[mask, "e3_aliases_new"] = match_group.loc[mask, "e3_aliases"].values
        
        rows.append(r2_g)

# %%
# concatenate all rows into a single dataframe

result = pd.concat(rows)

# %%
# validate the result

def check_errors(df):
    # idenfity self mapping rows
    self_mapping = (df["e2"] == df["e2_new"]) & (df["e3"] == df["e3_new"])

    # identify rows with no counterfactuals generated
    no_counterfactual = df["e2_new"].isna()

    failiure_cases = df[self_mapping | no_counterfactual]


    if not failiure_cases.empty:
        print(f"{self_mapping.sum()} self mapping cases found.")
        print(f"{no_counterfactual.sum()} rows with no counterfactuals generated.")
        print("Errors found in the following rows:")
        print(failiure_cases[['e1', 'e2', 'e3', 'e2_new', 'e3_new']].head())

        print(f"total rows: {len(df)}")
        print(f"total error rows: {len(failiure_cases)}")
        print(f"total valid rows: {len(df) - len(failiure_cases)}")
    else:
        print("No errors found in the result.")

check_errors(result)

# %%
# sort to ensure original order is maintained

result.sort_index(inplace=True)

#%%
# format and save the result

result.drop("Unnamed: 0", axis=1, inplace=True)
result.to_csv(args.output_path)
