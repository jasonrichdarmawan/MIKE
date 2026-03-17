# %%
# load dataset
import pandas as pd

df = pd.read_csv("../datasets/two_hop.csv")


# %%
# setup data structures for efficient lookup in main loop

# R1_Outs: r1 -> possible e2 outputs
# Used to find candidate B such that B is a possible output of R1
R1_Outs = (
    df.groupby("r1")["e2"]
    .apply(set)
    .to_dict()
)

# E2_Given_E1_R1: (e1, r1) -> existing e2 outputs
# Used to find candidate B such that B is not an existing output of R1 for (e1, r1)
E2_Given_E1_R1 = (
    df.groupby(["e1", "r1"])["e2"]
    .apply(set)
    .to_dict()
)

# E2_to_R2: e2 -> possible r2 values that take e2 as input
# all r2 for which r2(E) is defined for some E of type e2
E2_to_R2 = (
    df.groupby("e2")["r2"]
    .apply(set)
    .to_dict()
)

# e2_info: e2 -> {e2_label, e2_aliases}
# Used to get labels and aliases for the new e2 values
e2_info = (
    df.drop_duplicates("e2")
      .set_index("e2")[["e2_label", "e2_aliases", "e2_type"]]
      .to_dict("index")
)

# %%
# Main loop
# generate counterfactual e2 values

groups = df.groupby(["e1", "r1", "e2"], sort=False)

rows = []

for (E, R, b), g in groups:

    # record r2 values in GROUP
    R2_vals_in_group = set(g["r2"])

    # candidate B values
    candidates = R1_Outs.get(R, set()) - E2_Given_E1_R1.get((E, R), set())

    # maintain type consistency for e2
    orig_type = g["e2_type"].iloc[0]
    candidates = [
    B for B in candidates
        if e2_info.get(B, {}).get("e2_type") == orig_type
    ]

    best_B = None
    best_score = -1

    for B in candidates:

        R3_vals_for_B = E2_to_R2.get(B, set())

        # `score` is trying to pick a e2 entity that maximises the number of r2 connections
        # If there are a lot of r2 connections associated with some e2 entity Q ,
        # that will likely have a higher `score`
        # Hence `score` is biased towards more popular e2 entities.
        # It is also more biased to later entities (see if statement below)
        # Consequently number of unique e2_new = 552 while nunique e2 = 9177
        # TODO: find a fix? or see if this matters at all.
        score = len(R2_vals_in_group & R3_vals_for_B)

        # for two entities with equal scores the later one gets priority
        # because of "">=" so score is biased to later entities 
        if score >= best_score:
            best_score = score
            best_B = B

    if best_B is None:
        continue

    # copy() used so original df is not modified
    g = g.copy()
    
    # first in group should be edit row
    edit_id = g.iloc[0]["id"]

    # all other rows are propagated rows
    g["propagated_from"] = edit_id
    g["row_type"] = "propagated_eval"
    g.loc[g["id"] == edit_id, "propagated_from"] = pd.NA
    g.loc[g["id"] == edit_id, "row_type"] = "edit"

    g["e2_new"] = best_B

    info = e2_info.get(best_B, {"e2_label": None, "e2_aliases": None, "e2_type": None})

    g["e2_new_label"] = info["e2_label"]
    g["e2_new_aliases"] = info["e2_aliases"]
    g["e2_new_type"] = info["e2_type"]

    rows.append(g)

result = pd.concat(rows)


# %%
def add_e3_new():
    """Adding e3 values to the result dataframe"""

    # e3_lookup: (e2, r2) -> e3
    # map (e2, r2) to a set of values of e3
    e3_lookup = (
        df.drop_duplicates(["e2", "r2"])
        .set_index(["e2", "r2"])["e3"]
        .to_dict()
    )

    # e3_info: e3 -> {e3_label, e3_aliases}
    # Used to get labels and aliases for the new e3 values
    e3_info = (
        df.drop_duplicates("e3")
        .set_index("e3")[["e3_label", "e3_aliases", "e3_type"]]
        .to_dict("index")
    )


    # set e3_new based on the new e2 and existing r2 using the e3_lookup
    # TODO: if there are multiple e3 values, figure out how to deal with that....
    #      drop_duplicates() in e3_lookup leads to only first value being cosidered.
    result["e3_new"] = result.apply(
        lambda row: e3_lookup.get((row["e2_new"], row["r2"])),
        axis=1
    )

    # set e3_new_label, e3_new_aliases and e3_new_type based on e3_new using e3_info
    result["e3_new_label"] = result["e3_new"].apply(
        lambda e3: e3_info.get(e3, {"e3_label": None})["e3_label"])
    result["e3_new_aliases"] = result["e3_new"].apply(
        lambda e3: e3_info.get(e3, {"e3_aliases": None})["e3_aliases"])
    result["e3_new_type"] = result["e3_new"].apply(
        lambda e3: e3_info.get(e3, {"e3_type": None})["e3_type"])

add_e3_new()

# %%

# mark rows with no e2_new or no e3_new (for propagated rows) as invalid
result.loc[result["e2_new"].isna(), "row_type"] = "invalid"
result.loc[
    (result["row_type"] == "propagated_eval") & result["e3_new"].isna(),
    "row_type"
] = "invalid"

# seperate valid and invalid into two dataframes to help with putting invalid at the end
valid = result[result["row_type"] != "invalid"].copy()
invalid = result[result["row_type"] == "invalid"].copy()

# sort by edit (only valid)
valid["sort_group"] = valid["propagated_from"].fillna(valid["id"])
valid["is_edit"] = valid["row_type"] == "edit"

# sort by group first
# is is_edit is True it comes earliest in the group
# maintain id order for other rows
valid = valid.sort_values(
    by=["sort_group", "is_edit", "id"],
    ascending=[True, False, True]
)

result = pd.concat([valid, invalid], ignore_index=True)

# remove helper columns
result.drop(columns=["sort_group", "is_edit"], errors="ignore", inplace=True)

# %% 
# Basic tests to verify the quality of the generated counterfactuals

def test_no_overlap():
    """see if e2_new is different from e2 in all rows"""
    assert (result["e2_new"] == result["e2"]).sum() == 0

# make sure e3_new is consistent with other knowledge
def test_valid_r2():
    """see if (e2_new, r2, e3_new) are valid knowledge in previous dataset for all rows"""
    valid_knowledge = set(zip(df["e2"], df["r2"], df["e3"]))
    mask = result["e2_new"].notna() & result["e3_new"].notna()
    assert result.loc[mask].apply(lambda r: (r["e2_new"], r["r2"], r["e3_new"]) in valid_knowledge, axis=1).mean() == 1

def test_knowledge_consistency():
    """check if there are multiple e2_new values for the same (e1, r1, e2) group"""
    group_check = result.groupby(["e1","r1","e2"])["e2_new"].nunique()
    assert (group_check > 1).sum() == 0

def test_e2_type_consistency():
    """Verify that e2_new_type == e2_type for all non-null e2_new rows"""
    mask = result["e2_new"].notna()
    inconsistent = result.loc[mask & (result["e2_new_type"] != result["e2_type"])]
    assert len(inconsistent) == 0

def no_invalid_edit_rows():
    """check if there are any invalid rows with row_type == edit"""
    assert ((result["row_type"] == "edit") & (result["e2_new"].isna())).sum() == 0
    missing_e3_edit = result[
        (result["row_type"] == "edit") & (result["e3_new"].isna())
    ]
    print(f"{len(missing_e3_edit)} edit rows with missing e3_new values")

def test_all():
    test_no_overlap()
    test_valid_r2()
    test_knowledge_consistency()
    test_e2_type_consistency()
    no_invalid_edit_rows()

test_all()


# %%
# Stats about dataset
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
# find edit rows with no propagated rows

# all edit ids
edit_ids = set(result.loc[result["row_type"] == "edit", "id"])

# all ids that are used as sources for propagation
used_ids = set(result["propagated_from"].dropna())

# edits with NO propagation
no_propagation = edit_ids - used_ids

print(f"Edit rows with no propagation: {len(no_propagation)}")
print(f"Number of usable rows = {e3_new_count - len(no_propagation)}")

# %%
# Inspect overrepresentation of e2 and e3 values

print(f"Unique values in e2: {result['e2'].nunique()}")
print(f"Unique values in e2_new: {result['e2_new'].nunique()}")

print(f"Unique values in e3: {result['e3'].nunique()}")
print(f"Unique values in e3_new: {result['e3_new'].nunique()}")


# %%
def save_file():
    # format and save the result
    result.drop("Unnamed: 0", errors="ignore", axis=1, inplace=True)
    result.to_csv("../datasets/two_hop_cf_e2.csv")

save_file()