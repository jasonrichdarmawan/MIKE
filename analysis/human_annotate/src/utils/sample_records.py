import random

def sample_records(data, target_size, dataset_name: str):
    """
    Deterministically sample records with balanced (hop count, edit count)
    strata.
    """

    if target_size < 0:
        raise ValueError("target_size must be non-negative")

    match dataset_name:
        case "MQuAKE-CF-3k-v2":
            def make_stratum(record):
                return (
                    len(record["post"]["hop_wise"]),
                    len(record["requested_rewrite"]),
                )
        case "Two-Hop":
            def make_stratum(record):
                return (2, 1)
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    strata = {}
    for index, record in enumerate(data):
        strata.setdefault(make_stratum(record), []).append(index)

    rng = random.Random(20260826)
    for indices in strata.values():
        rng.shuffle(indices)

    # Round-robin allocation keeps strata balanced and naturally redistributes
    # slots when a stratum has fewer records than the others.
    chosen_indices = []
    while len(chosen_indices) < min(target_size, len(data)):
        added = False
        for stratum in sorted(strata):
            indices = strata[stratum]
            if indices:
                chosen_indices.append(indices.pop())
                added = True
                if len(chosen_indices) == target_size:
                    break
        if not added:
            break

    chosen_indices.sort()
    return [data[index] for index in chosen_indices]