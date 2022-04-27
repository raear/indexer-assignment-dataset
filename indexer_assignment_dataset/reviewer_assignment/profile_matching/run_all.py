from . import create_specter_document_embeddings
from . import create_specter_query_embeddings
from . import create_tfidf_document_embeddings
from . import create_tfidf_query_embeddings
from . import document_search
from . import pred


def main():
    create_tfidf_document_embeddings.main()
    create_tfidf_query_embeddings.main()
    create_specter_document_embeddings.main()
    create_specter_query_embeddings.main()
    document_search.main()
    pred.main()


if __name__ == "__main__":
    main()