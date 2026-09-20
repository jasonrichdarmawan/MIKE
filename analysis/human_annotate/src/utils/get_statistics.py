from collections import defaultdict
import logging

LOGGER = logging.getLogger(__name__)

def get_statistics(data, dataset_name: str):
    """
    Data statistics for the MQuAKE-CF-3k-v2 and counterfactual Two-Hop datasets.
    """
    match dataset_name:
        case "MQuAKE-CF-3k-v2":
            get_statistics_MQuAKE(data)
        case "Two-Hop":
            get_statistics_TwoHop(data)
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

def get_statistics_TwoHop(data):
    """
    Data statistics for the Two-Hop dataset.
    """
    LOGGER.info(f"Number of records: {len(data)}")

    edit_triples = [
        tuple((request["subject"], request["relation_id"]))
        for record in data
        for request in record["requested_rewrite"]
    ]
    LOGGER.info(f"Number of edit triples: {len(edit_triples)}")
    LOGGER.info(f"Number of unique edits: {len(set(edit_triples))}")

def get_statistics_MQuAKE(data):
    """
    Data statistics for the MQuAKE-CF-3k-v2 dataset.
    """
    LOGGER.info(f"Number of records: {len(data)}")

    edit_triples = [
        tuple((request["subject"], request["relation_id"]))
        for record in data
        for request in record["requested_rewrite"]
    ]
    LOGGER.info(f"Number of edit triples: {len(edit_triples)}")
    LOGGER.info(f"Number of unique edits: {len(set(edit_triples))}")

    record_count_by_hop = {
        hop: [
            record
            for record in data 
            if len(record["post"]["hop_wise"]) == hop
        ]
        for hop in range(2,5)
    }
    LOGGER.info("Number of records by hop count:")
    for hop, records in record_count_by_hop.items():
        LOGGER.info(f"  {hop} hops: {len(records)}")

    edit_count_by_hop = {
        count: [
            record 
            for record in data 
            if len(record["requested_rewrite"]) == count
        ]
        for count in range(1,5)
    }
    LOGGER.info("Number of records by edit count:")
    for count, records in edit_count_by_hop.items():
        LOGGER.info(f"  {count} edits: {len(records)}")

    # Strata
    strata = defaultdict(list)
    for record in data:
        hop_count = len(record["post"]["hop_wise"])
        edit_count = len(record["requested_rewrite"])
        strata[(hop_count, edit_count)].append(record)
    LOGGER.info("Number of records by (hop count, edit count):")
    for (hop_count, edit_count), records in sorted(strata.items()):
        LOGGER.info(f"  ({hop_count} hops, {edit_count} edits): {len(records)}")

    edited_subject_relation_count_by_record = {}
    for record in data:
        for request in record["requested_rewrite"]:
            subject_relation = (request["subject"], request["relation_id"])
            if subject_relation not in edited_subject_relation_count_by_record:
                edited_subject_relation_count_by_record[subject_relation] = set()
            edited_subject_relation_count_by_record[subject_relation].add(record["case_id"])
    LOGGER.info("Top 10 subject-relation pairs with the most edited records:")
    for subject_relation, records in sorted(edited_subject_relation_count_by_record.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        LOGGER.info(f"  {subject_relation}: {len(records)} records")