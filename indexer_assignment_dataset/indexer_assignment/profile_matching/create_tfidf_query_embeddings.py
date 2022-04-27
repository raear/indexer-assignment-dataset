from .config import CONFIG as cfg
import gzip
import json
import numpy as np
import pickle
import scipy.sparse
from ..shared.helper import enumerate_year_week_numbers, filter_dataset_by_weeknumber
from ...shared.helper import make_parent_dir


def main():
    vectorizer = pickle.load(open(cfg["tfidf_vectorizer_path"], "rb"))
    dataset = json.load(gzip.open(cfg["eval_dataset_path"], mode="rt", encoding="utf8"))

    for year, week_number in enumerate_year_week_numbers(cfg["eval_set_start_date"], cfg["eval_set_end_date"]):

        filtered_dataset = filter_dataset_by_weeknumber(dataset, year, week_number)
        if len(filtered_dataset) == 0:
            continue

        query_id_list = [example["pmid"] for example in filtered_dataset]
        query_text_list = [example["title"] + " " + example["abstract"] for example in filtered_dataset]

        query_ids = np.array(query_id_list, dtype=np.int).reshape([-1,1])
        query_embeddings = vectorizer.transform(query_text_list)

        query_ids_path = cfg["tfidf_query_ids_path_template"].format(year=year, week_number=week_number)
        tfidf_query_embeddings_path = cfg["tfidf_query_embeddings_path_template"].format(year=year, week_number=week_number)

        make_parent_dir(query_ids_path)

        np.save(query_ids_path, query_ids)
        scipy.sparse.save_npz(tfidf_query_embeddings_path, query_embeddings)


if __name__ == "__main__":
    main()