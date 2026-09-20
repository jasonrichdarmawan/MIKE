# %%

from argparse import ArgumentParser
import json
import sys

from src.prompt_to_messages import prompt_to_messages
from src.utils import render_messages

# %%

ARGV = (
    [
        "--model-name", "Qwen/Qwen2.5-7B-Instruct",
        # "--results-path", "/Users/jason/Library/Mobile Documents/com~apple~CloudDocs/20260915-MIKE-minor-revision/Source-Code/output/Qwen/Qwen2.5-7B-Instruct/two_hop_cf_e2/sequential_s1000_s32/Revised_AlphaEdit_20260724-161909",
        # "--dataset-name", "Two-Hop",
        "--results-path", "/Users/jason/Library/Mobile Documents/com~apple~CloudDocs/20260915-MIKE-minor-revision/Source-Code/output/Qwen/Qwen2.5-7B-Instruct/MQuAKE-CF-3k-v2-clke/sequential0_s1000_b32/Revised_AlphaEdit_20260726_093055",
        "--lang", "en",
        "--dataset-name", "MQuAKE-CF-3k-v2",
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
        "--lang",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
    )
    parser.add_argument(
        "--model-name",
        type=str,
    )
    args = parser.parse_args(argv)
    return args

args = parse_args(ARGV)

# %%

with open(f"{args.results_path}/{args.lang}/sampled_results.json", "r") as f:
    sampled_results = json.load(f)

# %%

metrics = {}
init = {
    "True positive": 0,
    "True negative": 0,
    "False positive": 0,
    "False negative": 0,
}
metrics.setdefault("Multi-hop question", init.copy())
metrics.setdefault("Multi-hop cloze", init.copy())
metrics.setdefault("Single-hop question", init.copy())
metrics.setdefault("Single-hop cloze", init.copy())

def compute_disagreement(sampled_results, dataset_name):
    for result in sampled_results:
        post = result['post']
        for result_column in ["hop_wise", "accuracy"]:
            generations_column = post[f"{result_column}_predictions"]
            automatic_judgements_column = post[result_column]
            human_judgements_column = post[f"{result_column}_human_judgement"]
            for question_index, (automatic_predictions, human_judgements, generations) in enumerate(
                zip(automatic_judgements_column, human_judgements_column, generations_column)
            ):
                # Hop-wise values are grouped by question, whereas accuracy values
                # contain one generation one prediction per question.
                is_grouped = isinstance(automatic_predictions, list)
                automatic_prediction_items = automatic_predictions if is_grouped else [automatic_predictions]
                human_judgement_items = human_judgements if is_grouped else [human_judgements]
                generation_items = generations if is_grouped else [generations]

                if dataset_name == "Two-Hop":
                    if not is_grouped:
                        if question_index == 0:
                            format_type = "cloze"
                        elif question_index == 1:
                            format_type = "question"
                        else:
                            raise ValueError(f"Unexpected question_index: {question_index}")
                elif dataset_name == "MQuAKE-CF-3k-v2":
                    if not is_grouped:
                        if question_index == 0:
                            format_type = "cloze"
                        elif question_index in [1, 2, 3]:
                            format_type = "question"
                        else:
                            raise ValueError(f"Unexpected question_index: {question_index}")
                else:
                    raise ValueError(f"Unknown dataset name: {dataset_name}")


                for item_index, (automatic_prediction, human_judgement, generation) in enumerate(
                    zip(automatic_prediction_items, human_judgement_items, generation_items)
                ):
                    if is_grouped:
                        if item_index == 0:
                            format_type = "cloze"
                        elif item_index == 1:
                            format_type = "question"
                        else:
                            raise ValueError(f"Unexpected item_index: {item_index}")

                    if result_column == "hop_wise":
                        hop_type = "Single-hop"
                    elif result_column == "accuracy":
                        hop_type = "Multi-hop"
                    else:
                        raise ValueError(f"Unexpected result_column: {result_column}")

                    column_key = f"{hop_type} {format_type}"

                    if automatic_prediction == human_judgement:
                        outcome = "True positive" if automatic_prediction else "True negative"
                    else:
                        outcome = "False positive" if automatic_prediction else "False negative"

                        print(f"{outcome} in case {result['case_id']}")

                        messages = prompt_to_messages(
                            prompt=generation["prompt"] + generation["generated"],
                            model_name=args.model_name,
                        )
                        print(f'Target: {generation["target"]}')
                        print(f"Automatic judgement: {automatic_prediction}")
                        print(f"Human judgement: {human_judgement}")
                        render_messages(messages)

                    metrics[column_key][outcome] += 1

compute_disagreement(sampled_results, args.dataset_name)

for column_key, counts in metrics.items():
    tp = counts["True positive"]
    tn = counts["True negative"]
    fp = counts["False positive"]
    fn = counts["False negative"]

    error = (fp + fn) / (tp + tn + fp + fn)

    print(f"Metrics for {column_key}:")
    print(f"  True positive: {tp}")
    print(f"  True negative: {tn}")
    print(f"  False positive: {fp}")
    print(f"  False negative: {fn}")
    print("   Error: {:.2f}%".format(error * 100))

print("Overall metrics:")
overall_tp = sum(counts["True positive"] for counts in metrics.values())
overall_tn = sum(counts["True negative"] for counts in metrics.values())
overall_fp = sum(counts["False positive"] for counts in metrics.values())
overall_fn = sum(counts["False negative"] for counts in metrics.values())
overall_error = (overall_fp + overall_fn) / (overall_tp + overall_tn + overall_fp + overall_fn)
print(f"  True positive: {overall_tp}")
print(f"  True negative: {overall_tn}")
print(f"  False positive: {overall_fp}")
print(f"  False negative: {overall_fn}")
print("   Overall Error: {:.2f}%".format(overall_error * 100))

# %%
