from .config import CONFIG as cfg
import json
from ...shared.helper import remove_ws


def main():
    reviewer_publications_list = json.load(open(cfg["reviewer_publications_with_metadata_path"]))
    with open(cfg["documents_path"], "wt") as docs_file:
        for reviewer_publications in reviewer_publications_list:
            reviewer_id = reviewer_publications["reviewer_id"] 
            for article in reviewer_publications["publications"]:
                doc_id = article["doc_id"]
                title = remove_ws(article["title"])
                abstract = remove_ws(article["abstract"])
                docs_file_line = f"{doc_id}\t{reviewer_id}\t{title}\t{abstract}\n"
                docs_file.write(docs_file_line)


if __name__ == "__main__":
    main()