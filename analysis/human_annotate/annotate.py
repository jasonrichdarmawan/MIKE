# %%

from argparse import ArgumentParser
import json
import logging
from logging_tee import setup_logger
from pathlib import Path
import sys

from src.utils import (
    get_statistics,
    sample_records,
)

from src.annotate_results import annotate_results

# %%

ARGV = (
    [
        "--model-name", "Qwen/Qwen2.5-7B-Instruct",
        # "--dataset-name", "Two-Hop",
        # "--results-path", "/Users/jason/Library/Mobile Documents/com~apple~CloudDocs/20260915-MIKE-minor-revision/Source-Code/output/Qwen/Qwen2.5-7B-Instruct/two_hop_cf_e2/sequential_s1000_s32/Revised_AlphaEdit_20260724-161909",
        "--dataset-name", "MQuAKE-CF-3k-v2",
        "--results-path", "/Users/jason/Library/Mobile Documents/com~apple~CloudDocs/20260915-MIKE-minor-revision/Source-Code/output/Qwen/Qwen2.5-7B-Instruct/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/Revised_AlphaEdit_20260726_093055",
        "--lang", "en",
    ]
    if "ipykernel" in sys.modules
    else None
)

# %%

def parse_args(argv: str | None = None):
    parser = ArgumentParser()
    parser.add_argument(
        "--results-path",
        type=str,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Two-Hop",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CohereLabs/aya-expanse-8b",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
    )
    return parser.parse_args(argv)

# %%

args = parse_args(ARGV)

# %%

results = {}
match args.dataset_name:
    case "Two-Hop":
        if lang != "en":
            raise ValueError("Two-Hop dataset only supports English")
    case "MQuAKE-CF-3k-v2":
        langs = ["en", "de", "fr", "ja", "zh"]
        if args.lang not in langs:
            raise ValueError(f"Invalid language for MQuAKE-CF-3k-v2: {args.lang}")
    case _:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

with open(f"{args.results_path}/en/metrics.json", "r") as f:
    results["en"] = json.load(f)
with open(f"{args.results_path}/{args.lang}/metrics.json", "r") as f:
    results[args.lang] = json.load(f)

# %%

log_file_path = f"{args.results_path}/{args.lang}/annotation_{args.start_from}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
setup_logger(
    log_file=log_file_path,
    level=logging.INFO,
)
print(f"Logging to {log_file_path}")

# %%

sample_en_file = Path(f"{args.results_path}/en/sampled_results.json")

if sample_en_file.exists():
    with open(sample_en_file, "r") as f:
        sample_en_results = json.load(f)
    sample_ids = [result["case_id"] for result in sample_en_results]
    print(f"Loaded sample IDs from {sample_en_file}")
else:
    if args.lang != "en":
        raise ValueError(f"Sample file not found for language {args.lang}: {sample_en_file}")

    sample_ids = [
        result["case_id"]
        for result in sample_records(
            results["en"],
            target_size=args.target_size,
            dataset_name=args.dataset_name,
        )
    ]
    print(f"Generated sample IDs: {sample_ids}")

# %%

sampled_results = []

sample_file = Path(f"{args.results_path}/{args.lang}/sampled_results.json")

if sample_file.exists():
    with open(sample_file, "r") as f:
        sampled_results = json.load(f)
    print(f"Loaded sampled results from {sample_file}")
    sample_ids_same = sample_ids == [result["case_id"] for result in sampled_results]
    print(f"Sample IDs match: {sample_ids_same}")
else:
    print(f"Sample file not found: {sample_file}")
    sampled_results = [
        result 
        for result in results[args.lang]
        if result["case_id"] in sample_ids
    ]

get_statistics(sampled_results, dataset_name=args.dataset_name)

# %%

for start_from in range(args.start_from, 20):
    end_at = start_from + 1
    annotate_results(sampled_results, model_name=args.model_name, start_from=start_from, end_at=end_at)

    save_path = f"{args.results_path}/{args.lang}/sampled_results.json"
    with open(save_path, "w") as f:
        json.dump(sampled_results, f, indent=2, ensure_ascii=False)
    print(f"Saved sampled results to {save_path}")

# %%