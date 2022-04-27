from .config import CONFIG as cfg
import gzip
import json


def save_qrels(dataset_path, qrels_path):
    dataset = json.load(gzip.open(dataset_path, mode="rt", encoding="utf8"))

    qrels = {}
    for example in dataset:
        pmid = example["pmid"]
        indexer_num = example["indexer_num"]
        qrels[pmid] = {}
        qrels[pmid][indexer_num] = 1

    with open(qrels_path, "wt") as qrels_file:
        for query_id in sorted(qrels):
            indexer_num_list = qrels[query_id]
            for indexer_num in sorted(indexer_num_list):
                line = f"{query_id}\tQ0\t{indexer_num}\t1\n"
                qrels_file.write(line)


def main():
    save_qrels(cfg["contractor_val_set_path"], cfg["contractor_val_set_qrels_path"])
    save_qrels(cfg["contractor_test_set_path"], cfg["contractor_test_set_qrels_path"])

    save_qrels(cfg["in_house_val_set_path"], cfg["in_house_val_set_qrels_path"])
    save_qrels(cfg["in_house_test_set_path"], cfg["in_house_test_set_qrels_path"])
    

if __name__ == "__main__":
    main()