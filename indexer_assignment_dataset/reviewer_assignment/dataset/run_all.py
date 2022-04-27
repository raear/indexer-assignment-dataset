from . import create_dataset
from . import create_documents_file
from . import create_exclude_list
from . import create_qrels_file
from . import create_queries_file
from . import create_reviewers_file
from . import create_train_test_sets
from . import download_dataset_article_metadata
from . import download_ftp_index
from . import download_pmc_data
from . import download_reviewer_publication_metadata
from . import download_reviewer_publications
from . import extract_article_authors
from . import extract_article_reviewer_pairs
from . import extract_xml_files


def main():
    download_ftp_index.main()
    download_pmc_data.main()
    extract_xml_files.main()
    extract_article_authors.main()
    extract_article_reviewer_pairs.main()
    download_reviewer_publications.main()
    download_reviewer_publication_metadata.main()
    create_dataset.main()
    download_dataset_article_metadata.main()
    create_train_test_sets.main()
    create_reviewers_file.main()
    create_queries_file.main()
    create_qrels_file.main()
    create_documents_file.main()
    create_exclude_list.main()


if __name__ == "__main__":
    main()