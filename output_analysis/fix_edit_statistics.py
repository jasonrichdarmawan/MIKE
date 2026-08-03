"""Correct fixed-target AlphaEdit optimization-step statistics in JSON files.

For AlphaEdit and fixed-target variants, the target is optimized only once per
request. Therefore, records for later edited layers 5--8 must report zero
optimization steps. Revised-AlphaEdit is excluded because it recomputes targets
after every layer update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ITEMS = [
	"output/Qwen/Qwen2.5-7B-Instruct/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_1-rank_compute-matched_20260725_192504/edit_statistics.json",
	"output/Qwen/Qwen2.5-7B-Instruct/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_20260726_161916/edit_statistics.json",
	"output/Qwen/Qwen2.5-7B-Instruct/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_compute-matched_20260724-162253/edit_statistics.json",
	"output/Qwen/Qwen2.5-7B-Instruct/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_20260724-194656/edit_statistics.json",
	"output/CohereLabs/aya-expanse-8b/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_1-rank_compute-matched_20260725_193934/edit_statistics.json",
	"output/CohereLabs/aya-expanse-8b/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_20260726_151737/edit_statistics.json",
	"output/CohereLabs/aya-expanse-8b/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_compute-matched_20260724-101725/edit_statistics.json",
	"output/CohereLabs/aya-expanse-8b/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_20260725-171804/edit_statistics.json",
	"output/CohereLabs/tiny-aya-global/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_1-rank_20260723_150949/edit_statistics.json",
	"output/CohereLabs/tiny-aya-global/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_1-rank_compute-matched_20260723_152313/edit_statistics.json",
	"output/CohereLabs/tiny-aya-global/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_20260723_183328/edit_statistics.json",
	"output/CohereLabs/tiny-aya-global/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_20260724-000254/edit_statistics.json",
	"output/CohereLabs/tiny-aya-global/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_compute-matched_20260724-063350/edit_statistics.json",
	"output/CohereLabs/tiny-aya-global/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_20260725-085458/edit_statistics.json",
	"output/Qwen/Qwen3-4B-Instruct-2507/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_1-rank_20260723_150845/edit_statistics.json",
	"output/Qwen/Qwen3-4B-Instruct-2507/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_1-rank_compute-matched_20260723_152241/edit_statistics.json",
	"output/Qwen/Qwen3-4B-Instruct-2507/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/AlphaEdit_20260723_152957/edit_statistics.json",
	"output/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_20260724-000058/edit_statistics.json",
	"output/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_1-rank_compute-matched_20260724-000600/edit_statistics.json",
	"output/Qwen/Qwen3-4B-Instruct-2507/two_hop_cf_e2/sequential_s1000_s32/AlphaEdit_20260724-211521/edit_statistics.json",
]

LAYERS_TO_FIX = {5, 6, 7, 8}
FIXED_TARGET_METHODS = {
	"AlphaEdit",
	"AlphaEdit_1-rank",
	"AlphaEdit_1-rank_compute-matched",
}


def fix_optimization_steps(path: Path, *, dry_run: bool = False) -> int:
	"""Set ``optimization_steps`` to zero for later fixed-target update rows."""
	with path.open(encoding="utf-8") as file:
		records = json.load(file)

	if not isinstance(records, list):
		raise ValueError(f"Expected a JSON array in {path}, but found {type(records).__name__}.")

	changed = 0
	for record in records:
		if (
			record.get("editing_method") in FIXED_TARGET_METHODS
			and record.get("edited_layer") in LAYERS_TO_FIX
			and record.get("optimization_steps") != 0
		):
			record["optimization_steps"] = 0
			changed += 1

	if dry_run:
		return changed

	# backup_path = path.with_suffix(path.suffix + ".bak")
	# if not backup_path.exists():
	# 	backup_path.write_bytes(path.read_bytes())

	with path.open("w", encoding="utf-8") as file:
		json.dump(records, file, indent=2)
		file.write("\n")

	return changed


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dry-run", action="store_true", help="Report the number of affected records without writing files."
	)
	args = parser.parse_args()
	
	for item in ITEMS:
		filepath = Path(item).resolve()
		changed = fix_optimization_steps(filepath, dry_run=args.dry_run)
		action = "Would correct" if args.dry_run else "Corrected"
		print(f"{action} {changed} fixed-target AlphaEdit record(s) in {filepath}.")
	# if not args.dry_run:
	# 	print(f"Backup: {filepath.with_suffix(filepath.suffix + '.bak')}")


if __name__ == "__main__":
	main()

# %%