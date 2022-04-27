from .config import CONFIG as cfg
import pandas as pd
import pickle
import scipy.sparse


def main():
    queries = pd.read_csv(cfg["queries_path"], sep="\t", header=None, dtype={1:str, 2:str}, keep_default_na=False)
    corpus = [ title + " " + abstract for title, abstract in zip(queries.iloc[:,1], queries.iloc[:,2]) ]
    
    vectorizer = pickle.load(open(cfg["tfidf_model_path"], "rb"))
    query_embeddings = vectorizer.transform(corpus)
    
    scipy.sparse.save_npz(cfg["tfidf_query_embeddings_path"], query_embeddings)


if __name__ == "__main__":
    main()