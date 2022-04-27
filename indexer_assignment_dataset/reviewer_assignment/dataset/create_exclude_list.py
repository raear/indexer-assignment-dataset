from .config import CONFIG as cfg
import json


def contruct_author_name(author_info):
    surname = author_info["surname"]
    surname = surname.strip().lower() if surname else ""
    given_names = author_info["given_names"]
    given_names = given_names.strip().lower() if given_names else ""
    first_name = given_names.split()[0].strip() if given_names else ""
    author_name = f"{surname} {first_name}"
    return author_name


def create_reviewer_by_name_lookup(reviewer_publications_list):
    lookup_reviewers_by_name = {}
    for reviewer in reviewer_publications_list:
        name = contruct_author_name(reviewer)
        if name not in lookup_reviewers_by_name:
            lookup_reviewers_by_name[name] = []
        lookup_reviewers_by_name[name].append(reviewer)
    return lookup_reviewers_by_name


def create_reviewer_by_orcid_lookup(reviewer_publications_list):
    lookup_reviewer_by_orcid =  {reviewer["orcid"]: reviewer for reviewer in reviewer_publications_list}
    return lookup_reviewer_by_orcid
    

def main():
    article_authors =            json.load(open(cfg["article_authors_path"]))
    dataset =                    json.load(open(cfg["dataset_with_metadata_path"]))
    reviewer_publications_list = json.load(open(cfg["reviewer_publications_with_metadata_path"]))

    lookup_reviewer_by_orcid =  create_reviewer_by_orcid_lookup(reviewer_publications_list)
    lookup_reviewers_by_name = create_reviewer_by_name_lookup(reviewer_publications_list)
   
    exclude_list = {}
    for article in dataset:
        pmid = article["pmid"]
        query_id = article["query_id"]
        reviewer_names = get_reviewer_names(article, lookup_reviewer_by_orcid)

        exclude_list[query_id] = []
        for author_info in article_authors[pmid]:
            orcid = author_info["orcid"]
            name = contruct_author_name(author_info)
            
            if orcid:
                if orcid in lookup_reviewer_by_orcid:
                    reviewer_id = lookup_reviewer_by_orcid[orcid]["reviewer_id"]
                    exclude_list[query_id].append(reviewer_id)
            elif name and name not in reviewer_names and name in lookup_reviewers_by_name:
                for reviewer in lookup_reviewers_by_name[name]:
                    reviewer_id = reviewer["reviewer_id"]
                    exclude_list[query_id].append(reviewer_id)
                    print(f'{query_id}: {author_info["given_names"]} {author_info["surname"]}: {reviewer["given_names"]} {reviewer["surname"]}')

    json.dump(exclude_list, open(cfg["exclude_list_path"], "wt"), indent=4, ensure_ascii=False)


def get_reviewer_names(article, lookup_reviewer_by_orcid):
    reviewer_names = set(contruct_author_name(lookup_reviewer_by_orcid[orcid]) for orcid in article["orcid_list"] if orcid in lookup_reviewer_by_orcid)
    return reviewer_names


if __name__ == "__main__":
    main()