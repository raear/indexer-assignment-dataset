from .config import CONFIG as cfg
import numpy as np
import pandas as pd
from ...shared.specter_embeddings_helper import create_embeddings


def main():
    queries = pd.read_csv(cfg["queries_path"], sep="\t", header=None, dtype={1:str, 2:str}, keep_default_na=False)
    data = { query_id: { "title": title, "abstract": abstract } for query_id, title, abstract in zip(queries.iloc[:,0], queries.iloc[:,1], queries.iloc[:,2])}
    _, query_embeddings = create_embeddings(data, cfg["specter_batch_size"], cfg["cache_dir"])
    np.save(cfg["specter_query_embeddings_path"], query_embeddings)


if __name__ == "__main__":
    main()