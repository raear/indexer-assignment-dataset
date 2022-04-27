from .config import CONFIG as cfg
import numpy as np
import os.path
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
from ..shared.helper import enumerate_year_week_numbers
from ...shared.helper import make_parent_dir, save_run


def main():
    search(cfg["tfidf_query_ids_path_template"], cfg["tfidf_query_embeddings_path_template"], cfg["tfidf_document_ids_path_template"], cfg["tfidf_document_embeddings_path_template"], cfg["tfidf_search_results_path_template"])
    search(cfg["specter_query_ids_path_template"], cfg["specter_query_embeddings_path_template"], cfg["specter_document_ids_path_template"], cfg["specter_document_embeddings_path_template"], cfg["specter_search_results_path_template"])


def search(query_ids_path_template, query_embeddings_path_template, document_ids_path_template, document_embeddings_path_template, search_results_path_template):

    for year, week_number in enumerate_year_week_numbers(cfg["eval_set_start_date"], cfg["eval_set_end_date"]):

        query_ids_path = query_ids_path_template.format(year=year, week_number=week_number)
        query_embeddings_path = query_embeddings_path_template.format(year=year, week_number=week_number)

        if not os.path.isfile(query_ids_path):
            continue

        extension = os.path.splitext(query_embeddings_path)[1]
        is_sparse = (extension.lower() == ".npz")

        query_ids = np.load(query_ids_path)
        if is_sparse:
            query_embeddings = scipy.sparse.load_npz(query_embeddings_path)
        else:
            query_embeddings = np.load(query_embeddings_path)

        for indexer_num in range(1, cfg["num_indexers"] + 1):
            qrels = {}
            search_results_path = search_results_path_template.format(year=year, week_number=week_number, indexer_num=indexer_num)
            print(f"Query: {query_embeddings_path}, Indexer Num: {indexer_num}")
            if os.path.isfile(search_results_path):
                continue
            
            document_ids_path = document_ids_path_template.format(indexer_num)
            document_embeddings_path = document_embeddings_path_template.format(indexer_num)
            
            document_ids = np.load(document_ids_path)
            if is_sparse:
                document_embeddings = scipy.sparse.load_npz(document_embeddings_path)
            else:
                document_embeddings = np.load(document_embeddings_path)

            distances = pairwise_distances(X=query_embeddings, Y=document_embeddings, metric="cosine", n_jobs=-1)
            
            for i, query_distances in enumerate(distances):
                query_id = query_ids[i,0].item()
                qrels[query_id] = {}

                top_indices = np.argsort(query_distances)[:cfg["search_top_n"]]
                top_query_distances = query_distances[top_indices]
                top_doc_ids = document_ids[top_indices,0]
                top_scores = 1. - top_query_distances / 2.
                for doc_id, score in zip(top_doc_ids, top_scores):
                    qrels[query_id][doc_id] = score

            make_parent_dir(search_results_path)
            save_run(search_results_path, qrels)


if __name__ == "__main__":
    main()