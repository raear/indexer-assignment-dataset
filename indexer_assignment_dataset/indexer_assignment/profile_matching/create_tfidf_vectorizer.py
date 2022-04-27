from .config import CONFIG as cfg
import gzip
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from ...shared.helper import make_parent_dir


def main():
    make_parent_dir(cfg["tfidf_vectorizer_path"])

    dataset = json.load(gzip.open(cfg["train_set_path"], mode="rt", encoding="utf8"))
    corpus = [ example["title"] + " " + example["abstract"] for example in dataset]
    del dataset
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    pickle.dump(vectorizer, open(cfg["tfidf_vectorizer_path"], "wb"))


if __name__ == "__main__":
    main()