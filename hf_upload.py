# %%

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

import sys

# %%

# uncomment if Run in a Jupyter notebook
# sys.argv = ["program.py"]

# %%

REPO_ID = "clke-results/CaKE"
REPO_TYPE = "dataset"
DEFAULT_FOLDER = Path("data/stats/tiny-aya-global/wikipedia_stats")
DEFAULT_PATH_IN_REPO = "jason/data/stats/tiny-aya-global/wikipedia_stats"

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Upload a folder to the clke-results/CaKE dataset repository."
	)
	parser.add_argument(
		"--folder",
		type=Path,
		default=DEFAULT_FOLDER,
		help=f"Local folder to upload (default: {DEFAULT_FOLDER})",
	)
	parser.add_argument(
		"--path-in-repo",
		default=DEFAULT_PATH_IN_REPO,
		help=f"Destination folder in the Hub repository (default: {DEFAULT_PATH_IN_REPO})",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	folder = args.folder.expanduser().resolve()

	if not folder.is_dir():
		raise SystemExit(f"Local folder does not exist or is not a directory: {folder}")

	# Opt in to the faster, resumable hf_xet transfer pipeline when available.
	os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

	api = HfApi()
	# api.create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
	commit_info = api.upload_folder(
		repo_id=REPO_ID,
		repo_type=REPO_TYPE,
		folder_path=folder,
		path_in_repo=args.path_in_repo.strip("/"),
		commit_message=f"Upload {args.path_in_repo.strip('/')}",
	)
	print(f"Upload complete: {commit_info.commit_url}")


if __name__ == "__main__":
	main()
# %%
