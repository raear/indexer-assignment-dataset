from .config import CONFIG as cfg
import json
import random


def main():
    dataset = json.load(open(cfg["dataset_with_metadata_path"]))

    test_set_fraction = cfg["test_set_fraction"]
    val_set_fraction  = cfg["val_set_fraction"]
    
    dataset_size = len(dataset)
    test_set_size = int(test_set_fraction*dataset_size)
    val_set_size = int(val_set_fraction*dataset_size)
    test_val_set_size = test_set_size + val_set_size

    random.shuffle(dataset)
    test_set = dataset[:test_set_size]
    val_set = dataset[test_set_size:test_val_set_size]
    train_set = dataset[test_val_set_size:]

    print(f"Dataset size: {dataset_size}")
    print(f"Test set size: {len(test_set)}")
    print(f"Val set size: {len(val_set)}")
    print(f"Train set size: {len(train_set)}")

    json.dump(test_set,  open(cfg["test_set_path"],  "wt"), indent=4, ensure_ascii=False)
    json.dump(val_set,   open(cfg["val_set_path"],   "wt"), indent=4, ensure_ascii=False)
    json.dump(train_set, open(cfg["train_set_path"], "wt"), indent=4, ensure_ascii=False)
   

if __name__ == "__main__":
    main()