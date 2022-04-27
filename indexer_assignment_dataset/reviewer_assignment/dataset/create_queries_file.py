from .config import CONFIG as cfg
import json
from ...shared.helper import remove_ws


def create_queries(input_dataset_path, queries_path):
    dataset = json.load(open(input_dataset_path))
    with open(queries_path, "wt") as queries_file:
        for example in dataset:
            query_id = example["query_id"]
            title = remove_ws(example["title"])
            abstract = remove_ws(example["abstract"])
            queries_file_line = f"{query_id}\t{title}\t{abstract}\n"
            queries_file.write(queries_file_line)

            
def main():
    create_queries(cfg["train_set_path"], cfg["train_queries_path"])
    create_queries(cfg["val_set_path"], cfg["val_queries_path"])
    create_queries(cfg["test_set_path"], cfg["test_queries_path"])


if __name__ == "__main__":
    main()