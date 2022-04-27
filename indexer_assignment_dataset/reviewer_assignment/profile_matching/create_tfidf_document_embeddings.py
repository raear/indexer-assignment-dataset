from .config import CONFIG as cfg
import pandas as pd
import pickle
import scipy.sparse
from ...shared.helper import make_parent_dir


def main():
    make_parent_dir(cfg["tfidf_document_embeddings_path"])

    documents = pd.read_csv(cfg["documents_path"], sep="\t", header=None, dtype={2:str, 3:str}, keep_default_na=False)
    corpus = [ title + " " + abstract for title, abstract in zip(documents.iloc[:,2], documents.iloc[:,3]) ]
    
    vectorizer = pickle.load(open(cfg["tfidf_model_path"], "rb"))
    document_embeddings = vectorizer.transform(corpus)

    scipy.sparse.save_npz(cfg["tfidf_document_embeddings_path"], document_embeddings)


if __name__ == "__main__":
    main()