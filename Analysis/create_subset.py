# %%

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# %%

dataset_dir = Path("../datasets")
model_name = "gpt2"

df = pd.read_csv(dataset_dir / model_name / "two_hop.csv")

# %%
"""
Caveat:
- if col "entity_shortcut_correct" is True, then filter out.
  - The goal is to filter out cases where no latent reasoning is performed. Specifically, where the model predicts due to a high correlation between :math:`e_` and :math:`e_3`.
- if col "relation_shortcut_correct" is True, then filter out.
  - The goal is to filter out cases where no latent reasoning is performed. Specifically, the model predicts generally popular entities.

Biran et al. [1] generate two dataset subsets per model:
- The first subset: The model correctly answers both the two-hop query and the first hop. Then, randomly sample 100 examples of each bridge entity type, in order to create a more balanced subset (if less than 100 examples of a specific type exist we take them all).
- The second subset: The model correctly answers both the first and the second hop in isolation, but fails to answer the full two-hop query. Then, randomly sample 50 examples of each bridge entity type (in order to keep the correct and incorrect subsets of approximately the same size).

Y. Yao et al. [2] focus on the second hop of reasoning, as the model typically perfoms well on the first hop. Y. Yao et al., [2] categorize the data from the HoppingTooLate dataset [1] into three subsets:
- Correct: The model answers both single-hop questions :math:`(e_1, r_1, e_2)` correctly, as well as the multi-hop question :math:`(e_1, r_1, r_2, ?)`.
- Inconsistent: The model answers both single-hop questions correctly but fails on the multi-hop question.
  - Y. Yao et al. [2] observe that some questions in the Correct set share the same 'brdige' entity :math:`e_2`, even though they originate from distinct subject-relation pairs, that the model answers correctly :math:`(e_1^', r_1^', r_2, ?)`. This suggests that while the model can leverage knowledge in some contexts, it fails to generalize, indicating reasoning gaps rather than missing knowledge.
- Incorrect: The model answers both single-hop questions correctly but fails on the multi-hop question in all contexts :math:`(e_1^', r_1^', e_2)`. This implies a complete failure to employ the knowledge for multi-hop reasoning.

Y. Yao et al. [2] conduct back-patch and cross-patch experiments, the source prompt and the target prompt are the same.

[1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
[2] Y. Yao et al., "CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners," in EMNLP 2025
"""

df_r = df[~df['entity_shortcut_correct'] & ~df['relation_shortcut_correct']]
df_r_correct = df_r[df_r['composition_correct'] & df_r['first_hop_correct'] & df_r['second_hop_correct']]
df_r_incorrect = df_r[~df_r['composition_correct'] & df_r['first_hop_correct'] & df_r['second_hop_correct']]

cols = ['e2_label', 'r2_type', 'e3_label']
correct_second_hop = []
for row in tqdm(df_r_correct.to_dict("records")):
    correct_second_hop.append((row['e2_label'], row['r2_type'], row['e3_label']))
correct_second_hop = set(correct_second_hop)

df_r_incorrect["is_correct_second_hop"] = df_r_incorrect.apply(
    lambda row: (row['e2_label'], row['r2_type'], row['e3_label']) in correct_second_hop,
    axis=1
)
df_r_inconsistent = df_r_incorrect[df_r_incorrect['is_correct_second_hop']]

# %%

def sample_group(group: pd.DataFrame, n=100):
    """
    If the number of samples in a group is less than n, then take all samples.
    """
    return group.sample(n=min(len(group), n), random_state=42)

df_r_correct_sampled = df_r_correct.groupby(["r1_type", "r2_type"], group_keys=False).apply(sample_group)
df_r_correct_sampled.to_csv(dataset_dir / model_name / "correct.csv")
print(f"Sampled df_r_correct from {len(df_r_correct)} to {len(df_r_correct_sampled)}")

df_r_incorrect_sampled = df_r_incorrect[~df_r_incorrect.index.isin(df_r_inconsistent.index)].groupby(["r1_type", "r2_type"], group_keys=False).apply(sample_group, n=50)
df_r_incorrect_sampled.to_csv(dataset_dir / model_name / "incorrect.csv")
print(f"Sampled df_r_incorrect from {len(df_r_incorrect)} to {len(df_r_incorrect_sampled)}")

df_r_inconsistent = df_r_inconsistent.groupby(["r1_type", "r2_type"], group_keys=False).apply(sample_group, n=30)
df_r_inconsistent.to_csv(dataset_dir / model_name / "inconsistent.csv")
print(f"Sampled df_r_inconsistent from {len(df_r_inconsistent)} to {len(df_r_inconsistent)}")

# %%