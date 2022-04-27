from .config import CONFIG as cfg
import json


def main():
    reviewer_publications_list = json.load(open(cfg["reviewer_publications_with_metadata_path"]))
    with open(cfg["reviewers_path"], "wt") as reviewers_file:
        for reviewer_publications in reviewer_publications_list:
            reviewer_id = reviewer_publications["reviewer_id"]
            orcid = reviewer_publications["orcid"].strip()
            surname = reviewer_publications["surname"].strip()
            given_names = reviewer_publications["given_names"].strip()
            reviewers_file_line = f"{reviewer_id}\t{orcid}\t{surname}\t{given_names}\n"
            reviewers_file.write(reviewers_file_line)


if __name__ == "__main__":
    main()