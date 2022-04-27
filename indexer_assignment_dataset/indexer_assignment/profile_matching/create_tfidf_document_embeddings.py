from .config import CONFIG as cfg
from .helper import create_indexer_history_lookup
import numpy as np
import pickle
import scipy.sparse
from ...shared.helper import make_parent_dir


def main():
    tfidf_document_ids_path_ex = cfg["tfidf_document_ids_path_template"].format(1)
    make_parent_dir(tfidf_document_ids_path_ex)

    vectorizer = pickle.load(open(cfg["tfidf_vectorizer_path"], "rb"))
    indexer_history_lookup = create_indexer_history_lookup(cfg["train_set_path"])
    
    for indexer_num in sorted(indexer_history_lookup):
        indexer_history = indexer_history_lookup[indexer_num]

        doc_id_list = []
        doc_text_list = []
        for doc_id, doc_text_dict in indexer_history.items():
            doc_text = doc_text_dict["title"] + " " + doc_text_dict["abstract"] 
            doc_id_list.append(doc_id)
            doc_text_list.append(doc_text)

        document_ids = np.array(doc_id_list, dtype=np.int).reshape([-1,1])
        document_embeddings = vectorizer.transform(doc_text_list)
    
        tfidf_document_ids_path = cfg["tfidf_document_ids_path_template"].format(indexer_num)
        tfidf_document_embeddings_path = cfg["tfidf_document_embeddings_path_template"].format(indexer_num)

        np.save(tfidf_document_ids_path, document_ids)
        scipy.sparse.save_npz(tfidf_document_embeddings_path, document_embeddings)
  

if __name__ == "__main__":
    main()