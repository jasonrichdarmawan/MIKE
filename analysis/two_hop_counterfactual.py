# generate counterfactuals for two-hop dataset 
#   and saves them in `two_hop_counterfactual.csv`
# for each row with r1==r1, but e1!=e1 
#   swap r2 and r3 values to generate counterfactuals
# %%
import pandas as pd

df = pd.read_csv("../datasets/two_hop.csv")

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

print("Errors before fix:")
check_errors(result)

# 34767 rows with no counterfactuals generated

# %%
# Brute Force by grouping by r2 
# Takes around 10mins to run on 34767 rows
# earlier rows are overrepresented as e2_new and e3_new values 

missing_mask = result["e2_new"].isna() & result["e3_new"].isna()

# For rows that could not be matched within their (r1, r2) group, search across
# all rows with the same r2 and copy the first valid non-self counterfactual.
for idx, row in result[missing_mask].iterrows():
    # group by both r1 and r2
    # same_r1_r2 = result[(result["r2"] == row["r2"]) & (result["r1"] == row["r1"]) &(result.index != idx)]
    # Takes longer to run so we keep shifting logic from earlier instead of this bruteforce method
    same_r2 = result[(result["r2"] == row["r2"]) & (result.index != idx)]

    replacement = None
    for _, candidate in same_r2.iterrows():
        # see if candidate row has valid data
        candidate_valid = (
            pd.notna(candidate["e2_new"])
            and pd.notna(candidate["e3_new"])
            and candidate["e2"] != candidate["e2_new"]
        )

        # Ensure non-self mapping
        target_non_self = (
            row["e2"] != candidate["e2_new"]
            or row["e3"] != candidate["e3_new"]
        )

        if candidate_valid and target_non_self:
            replacement = candidate
            break

    if replacement is not None:
        result.at[idx, "e2_new"] = replacement["e2_new"]
        result.at[idx, "e3_new"] = replacement["e3_new"]
        result.at[idx, "e2_label_new"] = replacement["e2_label_new"]
        result.at[idx, "e3_label_new"] = replacement["e3_label_new"]
        result.at[idx, "e2_aliases_new"] = replacement["e2_aliases_new"]
        result.at[idx, "e3_aliases_new"] = replacement["e3_aliases_new"]


# %%
print("Errors after fix:")
check_errors(result)

# No errors found

# %%
# sort to ensure original order is maintained
result.sort_index(inplace=True)

# format and save the result
result.drop("Unnamed: 0", axis=1, inplace=True)
result.to_csv("../datasets/two_hop_counterfactual.csv")
