# %%

import json
from pathlib import Path

import pandas as pd

# %%

ROOT = Path("./output/Qwen/Qwen3-4B-Instruct-2507/MQuAKE-CF-3k-v2-clke/sequential_s1000_b32")
METHOD_FOLDERS = [
	"R-ROME_sequential",
	"WISE_sequential_eval_batch",
	"CaKE_sequential",
	"AlphaEdit_Circuit_sequential",
]
BENCHMARK_KEY = "nli"

rows = []
for method_folder in METHOD_FOLDERS:
	method_name = method_folder.replace("_sequential", "").replace("_eval_batch", "")
	for json_file in sorted((ROOT / method_folder / "glue_eval").glob("case_*_glue.json")):
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

pivot.to_clipboard(sep="\t", excel=True)
print(f"Copied {BENCHMARK_KEY.upper()} f1_new table to clipboard. Shape: {pivot.shape}")

# %%
