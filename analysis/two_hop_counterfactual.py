# generate counterfactuals for two-hop dataset 
#   and saves them in `two_hop_counterfactual.csv`
# for each row with r1==r1, but e1!=e1 
#   swap r2 and r3 values to generate counterfactuals
# %%
import pandas as pd

df = pd.read_csv("../datasets/two_hop.csv")

# group dataframe by r1
groups = df.groupby("r1")

# %%
def shift_generate():
    # use rows with same r1 to generate counterfactuals by swapping r2 and r3 values
    rows = []

    for _, g in groups:
        # shift the group by one and 
        shifted = g.shift(-1)
        shifted.iloc[-1] = g.iloc[0]

        mask = g["e1"] != shifted["e1"]

        g = g.loc[mask].copy()
        shifted = shifted.loc[mask]

        g["e2_new"] = shifted["e2"]
        g["e3_new"] = shifted["e3"]

        g["e2_aliases_new"] = shifted["e2_aliases"]
        g["e3_aliases_new"] = shifted["e3_aliases"]

        rows.append(g)

    result = pd.concat(rows)
    return result


# %%
def shuffle_generate():
    # instead of shifting, shuffle group
    rows = []

    for _, g in groups:
        # randomize rows inside the group instead of shifting
        g = g.reset_index(drop=True)
        shuffled = g.sample(frac=1).reset_index(drop=True)

        mask = g["e1"] != shuffled["e1"]

        g = g.loc[mask].copy()
        shuffled = shuffled.loc[mask]

        g["e2_new"] = shuffled["e2"]
        g["e3_new"] = shuffled["e3"]

        g["e2_aliases_new"] = shuffled["e2_aliases"]
        g["e3_aliases_new"] = shuffled["e3_aliases"]

        rows.append(g)

    result = pd.concat(rows)
    return result

# %%
result = shuffle_generate()
result.drop("Unnamed: 0", axis=1, inplace=True)
result.to_csv("two_hop_counterfactual.csv", index=False)
