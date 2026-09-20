from ..utils import render_messages
from ..prompt_to_messages import prompt_to_messages

ANNOTATION_COLUMNS = (
    ("hop_wise_predictions", "hop_wise"),
    ("accuracy_predictions", "accuracy"),
)


def empty_judgements_like(predictions):
    """Create a nested list of ``None`` values matching the predictions."""
    if isinstance(predictions, list):
        return [empty_judgements_like(item) for item in predictions]
    return None


def ask_for_judgement() -> bool:
    """Prompt until the annotator enters a valid yes/no response."""
    while True:
        answer = input("Is the generated answer correct? (y/n): ").strip().lower()
        if answer in {"y", "n"}:
            return answer == "y"
        print("Invalid input. Please enter 'y' or 'n'.")


def annotate_predictions(post: dict, generation_column: str, prediction_column: str, model_name: str):
    """Collect human labels for every generation in one prediction category."""
    predictions = post[prediction_column]
    generations = post[generation_column]
    judgement_column = f"{prediction_column}_human_judgement"
    judgements = post.setdefault(
        judgement_column,
        empty_judgements_like(predictions),
    )

    for question_index, (question_generations, automatic_predictions) in enumerate(
        zip(generations, predictions)
    ):
        # Hop-wise values are grouped by question, whereas accuracy values
        # contain one generation and one prediction per question.
        is_grouped = isinstance(question_generations, list)
        generation_items = question_generations if is_grouped else [question_generations]
        prediction_items = automatic_predictions if is_grouped else [automatic_predictions]
        judgement_items = judgements[question_index] if is_grouped else [judgements[question_index]]

        for item_index, (generation_item, automatic_prediction) in enumerate(
            zip(generation_items, prediction_items)
        ):
            if judgement_items[item_index] is not None:
                continue

            messages = prompt_to_messages(
                prompt=generation_item["prompt"] + generation_item["generated"],
                model_name=model_name,
            )
            print(f'Target: {generation_item["target"]}')
            print(f"Automatic judgement: {automatic_prediction}")
            render_messages(messages)

            judgement = ask_for_judgement()
            print(f"Human judgement: {judgement}")
            if is_grouped:
                judgements[question_index][item_index] = judgement
            else:
                judgements[question_index] = judgement


def annotate_results(sampled_records: list[dict], model_name: str, start_from: int = 0, end_at: int | None = None):
    """Collect human judgments for sampled records, starting at an offset."""
    for sample_index, result in enumerate(
        sampled_records[start_from:end_at],
        start=start_from,
    ):
        print(f"Annotating sample {sample_index + 1} of {len(sampled_records)}")
        for generation_column, prediction_column in ANNOTATION_COLUMNS:
            annotate_predictions(result["post"], generation_column, prediction_column, model_name)