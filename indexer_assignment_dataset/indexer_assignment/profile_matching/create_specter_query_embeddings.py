from .config import CONFIG as cfg
import gzip
import json
import numpy as np
from ..shared.helper import enumerate_year_week_numbers, filter_dataset_by_weeknumber
from ...shared.helper import make_parent_dir
from ...shared.specter_embeddings_helper import create_embeddings


def main():
    dataset = json.load(gzip.open(cfg["eval_dataset_path"], mode="rt", encoding="utf8"))

    for year, week_number in enumerate_year_week_numbers(cfg["eval_set_start_date"], cfg["eval_set_end_date"]):

        filtered_dataset = filter_dataset_by_weeknumber(dataset, year, week_number)
        if len(filtered_dataset) == 0:
            continue
        data = { example["pmid"]: { "title": example["title"], "abstract": example["abstract"] } for example in filtered_dataset }
        query_ids, query_embeddings = create_embeddings(data, cfg["specter_batch_size"], cfg["cache_dir"])
        
        specter_query_ids_path = cfg["specter_query_ids_path_template"].format(year=year, week_number=week_number)
        specter_query_embeddings_path = cfg["specter_query_embeddings_path_template"].format(year=year, week_number=week_number)

        make_parent_dir(specter_query_ids_path)

        np.save(specter_query_ids_path, query_ids)
        np.save(specter_query_embeddings_path, query_embeddings)


if __name__ == "__main__":
    main()