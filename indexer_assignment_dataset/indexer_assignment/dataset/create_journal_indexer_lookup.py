from .config import CONFIG as cfg
import gzip
import json
import pickle


def create_lookup(train_set_path, val_set_path, test_set_path, save_path):
    
    train_set = json.load(gzip.open(train_set_path, "rt", encoding="utf8"))
    val_set = json.load(gzip.open(val_set_path, "rt", encoding="utf8"))
    dataset = train_set + val_set
    if test_set_path:
        test_set = json.load(gzip.open(test_set_path, "rt", encoding="utf8"))
        dataset = dataset + test_set

    journal_indexer_lookup = {}
    for example in dataset:
        nlmid = example["journal_nlmid"]
        indexer_num = example["indexer_num"]
        if nlmid not in journal_indexer_lookup:
            journal_indexer_lookup[nlmid] = set()
        journal_indexer_lookup[nlmid].add(indexer_num)
        
    pickle.dump(journal_indexer_lookup, open(save_path, "wb"))
    

def main():
    create_lookup(cfg["contractor_train_set_path"], cfg["contractor_val_set_path"], None, cfg["contractor_journal_indexer_lookup_path"])
    create_lookup(cfg["in_house_train_set_path"],   cfg["in_house_val_set_path"],   None,  cfg["in_house_journal_indexer_lookup_path"])

    create_lookup(cfg["contractor_train_set_path"], cfg["contractor_val_set_path"], cfg["contractor_test_set_path"], cfg["contractor_eval_journal_indexer_lookup_path"])
    create_lookup(cfg["in_house_train_set_path"],   cfg["in_house_val_set_path"],   cfg["in_house_test_set_path"],   cfg["in_house_eval_journal_indexer_lookup_path"])


if __name__ == "__main__":
    main()