from .config import CONFIG as cfg
import pickle


def create_lookup(journal_indexer_lookup_path, indexer_journal_descriptor_lookup_path):
    journal_descriptor_lookup = pickle.load(open(cfg["journal_descriptor_lookup_path"], "rb"))
    journal_indexer_lookup =    pickle.load(open(journal_indexer_lookup_path, "rb"))
    
    lookup = {}
    for journal_id in journal_indexer_lookup:
        if journal_id not in journal_descriptor_lookup:
            print(journal_id)
            continue
        journal_descriptors = journal_descriptor_lookup[journal_id] 
        for indexer_num in journal_indexer_lookup[journal_id]:
            if indexer_num not in lookup:
                lookup[indexer_num] = set()
            lookup[indexer_num].update(journal_descriptors)
            
    pickle.dump(lookup, open(indexer_journal_descriptor_lookup_path, "wb"))


def main():
    create_lookup(cfg["contractor_eval_journal_indexer_lookup_path"], cfg["contractor_indexer_journal_descriptor_lookup_path"])
    create_lookup(cfg["in_house_eval_journal_indexer_lookup_path"], cfg["in_house_indexer_journal_descriptor_lookup_path"])
    

if __name__ == "__main__":
    main()