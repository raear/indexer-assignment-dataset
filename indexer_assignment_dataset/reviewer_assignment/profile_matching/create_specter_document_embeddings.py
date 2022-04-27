from .config import CONFIG as cfg
import numpy as np
import pandas as pd
from ...shared.specter_embeddings_helper import create_embeddings


def main():
    documents = pd.read_csv(cfg["documents_path"], sep="\t", header=None, dtype={2:str, 3:str}, keep_default_na=False)
    data = { doc_id: { "title": title, "abstract": abstract } for doc_id, title, abstract in zip(documents.iloc[:,0], documents.iloc[:,2], documents.iloc[:,3])}
    _, doc_embeddings = create_embeddings(data, cfg["specter_batch_size"], cfg["cache_dir"])
    np.save(cfg["specter_document_embeddings_path"], doc_embeddings)


if __name__ == "__main__":
    main()