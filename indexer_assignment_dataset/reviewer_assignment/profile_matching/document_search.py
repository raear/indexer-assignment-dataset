from .config import CONFIG as cfg
import numpy as np
import os.path
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
from ...shared.helper import save_run


def search(document_embeddings_path, query_embeddings_path, search_results_path):
    queries_data = pd.read_csv(cfg["queries_path"], sep="\t", header=None, dtype={1:str, 2:str}, keep_default_na=False)
    query_ids = list(queries_data.iloc[:,0])
    
    extension = os.path.splitext(query_embeddings_path)[1]
    is_sparse = (extension.lower() == ".npz")

    if is_sparse:
        query_embeddings = scipy.sparse.load_npz(query_embeddings_path)
        document_embeddings = scipy.sparse.load_npz(document_embeddings_path)
    else:
        query_embeddings = np.load(query_embeddings_path)
        document_embeddings = np.load(document_embeddings_path)

    distances = pairwise_distances(X=query_embeddings, Y=document_embeddings, metric="cosine")

    qrels = {}
    for i, query_distances in enumerate(distances):
        query_id = query_ids[i]
        qrels[query_id] = {}

        top_indices = np.argsort(query_distances)[:cfg["search_top_n"]]
        top_query_distances = query_distances[top_indices]
        top_doc_ids = top_indices + 1
        top_scores = 1. - top_query_distances / 2.
        for doc_id, score in zip(top_doc_ids, top_scores):
            qrels[query_id][doc_id] = score

    save_run(search_results_path, qrels)
    

def main(): 
    search(cfg["tfidf_document_embeddings_path"], cfg["tfidf_query_embeddings_path"], cfg["tfidf_search_results_path"])
    search(cfg["specter_document_embeddings_path"], cfg["specter_query_embeddings_path"], cfg["specter_search_results_path"])


if __name__ == "__main__":
    main()