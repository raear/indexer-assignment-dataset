from .config import CONFIG as cfg
from .helper import create_indexer_history_lookup
import numpy as np
from ...shared.helper import make_parent_dir
from ...shared.specter_embeddings_helper import create_embeddings


def main():
    save_path_ex = cfg["specter_document_ids_path_template"].format(1)
    make_parent_dir(save_path_ex)

    indexer_history_lookup = create_indexer_history_lookup(cfg["train_set_path"])
    for indexer_num in sorted(indexer_history_lookup):
        indexer_history = indexer_history_lookup[indexer_num]
        doc_ids, doc_embeddings = create_embeddings(indexer_history, cfg["specter_batch_size"], cfg["cache_dir"])
        ids_save_path = cfg["specter_document_ids_path_template"].format(indexer_num)
        docs_save_path = cfg["specter_document_embeddings_path_template"].format(indexer_num)
        np.save(ids_save_path, doc_ids)
        np.save(docs_save_path, doc_embeddings)


if __name__ == "__main__":
    main()