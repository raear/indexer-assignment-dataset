from .config import CONFIG as cfg
import gzip
from ...shared.helper import to_date
import json
import random


def split(dataset_path, train_path, val_path, test_path):

    dataset = json.load(gzip.open(dataset_path, "rt", encoding="utf8"))

    indexed_data = [example for example in dataset if example["date_completed"] is not None]
    del dataset

    indexer_examples = {}
    for example in indexed_data:
        indexer_id = example["indexer_id"]
        example["date_completed_date"] = to_date(example["date_completed"])
        if indexer_id not in indexer_examples:
            indexer_examples[indexer_id] = []
        indexer_examples[indexer_id].append(example)
    del indexed_data

    sorted_current_indexer_examples = {}
    for indexer_id in indexer_examples:
        examples = indexer_examples[indexer_id]
        sorted_examples = sorted(examples, key=lambda x: x["date_completed_date"])
        first_example = sorted_examples[0]
        last_example = sorted_examples[-1]
        if (first_example["date_completed_date"] <= cfg["new_indexer_cutoff_date"] and
            last_example["date_completed_date"] >= cfg["test_set_start_date"]):
            sorted_current_indexer_examples[indexer_id] = sorted_examples
    del indexer_examples

    def _transform(example_list, indexer_num):
        for example in example_list:
            del example["date_completed_date"]
            example["indexer_num"] = indexer_num
        return example_list

    def _sort_func(example):
        date_completed_date = example["date_completed_date"]
        medline_ta = example["medline_ta"]
        volume = str(example["volume"])
        issue = str(example["issue"])
        pmid = int(example["pmid"])
        volume_int = int(volume) if volume.isnumeric() else -1
        issue_int =  int(issue)  if issue.isnumeric()  else -1
        return date_completed_date, medline_ta, volume_int, issue_int, pmid

    train_set = []
    val_set = []
    test_set = []
    for idx, indexer_id in enumerate(sorted(sorted_current_indexer_examples)):
        indexer_num = idx + 1
        indexer_examples = sorted_current_indexer_examples[indexer_id]
        sorted_examples = sorted(indexer_examples, key=_sort_func)
        indexer_train_set_citations = []
        indexer_val_set_citations = []
        indexer_test_set_citations = []
        for example in sorted_examples:
            date_completed_date = example["date_completed_date"]
            if date_completed_date < cfg["val_set_start_date"]:
                indexer_train_set_citations.append(example)
            elif date_completed_date >= cfg["val_set_start_date"] and date_completed_date < cfg["test_set_start_date"]:
                indexer_val_set_citations.append(example)
            elif date_completed_date >= cfg["test_set_start_date"] and date_completed_date <= cfg["test_set_end_date"]:
                indexer_test_set_citations.append(example)
        indexer_train_set_citations = _transform(indexer_train_set_citations, indexer_num)
        train_set.extend(indexer_train_set_citations)
        indexer_val_set_citations = _transform(indexer_val_set_citations, indexer_num)
        val_set.extend(indexer_val_set_citations)
        indexer_test_set_citations = _transform(indexer_test_set_citations, indexer_num)
        test_set.extend(indexer_test_set_citations)
    del sorted_current_indexer_examples
            
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    save_dataset(train_path, train_set)
    save_dataset(val_path, val_set)
    save_dataset(test_path, test_set)


def main():
    split(cfg["contractor_dataset_with_metadata_path"], cfg["contractor_train_set_path"], cfg["contractor_val_set_path"], cfg["contractor_test_set_path"])
    split(cfg["in_house_dataset_with_metadata_path"], cfg["in_house_train_set_path"], cfg["in_house_val_set_path"], cfg["in_house_test_set_path"])
    

def save_dataset(path, dataset):
    with gzip.open(path, mode="wt", encoding="utf8") as write_file:
        write_file.write("[")
        is_first = True
        for example in dataset:
            if is_first:
                is_first = False
            else:
                write_file.write(",\n")
            example_str = json.dumps(example, ensure_ascii=False)
            write_file.write(example_str)
        write_file.write("]")


if __name__ == "__main__":
    main()