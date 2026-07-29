# %%

import json
from pathlib import Path

import pandas as pd

# %%

ROOT = Path("./output/CohereLabs/aya-expanse-8b/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32")
METHOD_FOLDERS = [
	"AlphaEdit_20260726_151737",
	"Revised_AlphaEdit_20260726_093849",
]
BENCHMARK_KEY = "nli"

rows = []
for method_folder in METHOD_FOLDERS:
	method_name = method_folder
	for json_file in sorted((ROOT / method_folder / "glue_eval").glob("case_*_glue.json")):
		print(f"Processing {json_file}")
		with json_file.open("r", encoding="utf-8") as f:
			data = json.load(f)

		f1_new = data.get(BENCHMARK_KEY, {})["f1_new"]
		edit_num = data["edit_num"]

		if f1_new is not None and edit_num is not None:
			rows.append(
				{
					"Method Name": method_name,
					"Edit Num": int(edit_num),
					f"{BENCHMARK_KEY.upper()}": f1_new,
				}
			)


df = pd.DataFrame(rows)
pivot = df.pivot(index="Method Name", columns="Edit Num", values=BENCHMARK_KEY.upper()).sort_index(axis=1)

pivot.to_clipboard(sep=",", excel=True)
print(f"Copied {BENCHMARK_KEY.upper()} f1_new table to clipboard. Shape: {pivot.shape}")

# %%
