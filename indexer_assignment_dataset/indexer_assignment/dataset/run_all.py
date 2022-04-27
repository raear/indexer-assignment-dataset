from . import add_dataset_metadata
from . import create_eval_qrels
from . import create_indexer_journal_descriptor_lookup
from . import create_issue_id_lookup
from . import create_journal_descriptor_lookup
from . import create_journal_indexer_lookup
from . import download_medline_data
from . import extract_citation_data
from . import split_datsets


def main():
    download_medline_data.main()
    extract_citation_data.main()
    add_dataset_metadata.main()
    split_datsets.main()
    create_eval_qrels.main()
    create_journal_indexer_lookup.main()
    create_issue_id_lookup.main()
    create_journal_descriptor_lookup.main()
    create_indexer_journal_descriptor_lookup.main()
  
  
if __name__ == "__main__":
    main()