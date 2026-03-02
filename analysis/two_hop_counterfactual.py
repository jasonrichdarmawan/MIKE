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
        
        # add e2_new and e3_new only where both r2 and r3 match
        mask = (r2_g["r1"] == match_group["r1"]) & (r2_g["r2"] == match_group["r2"])
        
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

# sort to ensure original order is maintained
result.sort_index(inplace=True)


#%%
# format and save the result
result.drop("Unnamed: 0", axis=1, inplace=True)
result.to_csv("../datasets/two_hop_counterfactual.csv")
