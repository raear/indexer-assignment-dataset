import json
import gzip


def create_indexer_history_lookup(dataset_path):
    dataset = json.load(gzip.open(dataset_path, mode="rt", encoding="utf8"))
    indexer_history_lookup = {}
    for example in dataset:
        indexer_num = example["indexer_num"]
        if indexer_num not in indexer_history_lookup:
            indexer_history_lookup[indexer_num] = {}
        doc_id = example["pmid"]
        indexer_history_lookup[indexer_num][doc_id] = { "title": example["title"], "abstract": example["abstract"]}
    return indexer_history_lookup