from .config import CONFIG as cfg
import json
import pandas as pd


def main():
    
    pairs = pd.read_csv(cfg["article_reviewer_pairs_path"], sep=",", header=None, dtype={3:str}, keep_default_na=False)

    reviewer_publications_list = json.load(open(cfg["reviewer_publications_with_metadata_path"]))
    lookup_reviewer_by_orcid =  {reviewer["orcid"]: reviewer for reviewer in reviewer_publications_list}

    query_id = 0
    dataset = {}
    for _, (surname, given_names, orcid, pmid, pmcid) in list(pairs.iterrows())[:]:
        
        if not orcid or pd.isna(orcid):
            continue

        orcid = str(orcid)
        if not orcid in lookup_reviewer_by_orcid:
            continue
        
        reviewer = lookup_reviewer_by_orcid[orcid]
        reviewer_id = reviewer["reviewer_id"]

        if pmid not in dataset:
            query_id += 1
            dataset[pmid] = { "pmid": pmid, "query_id": query_id, "orcid_list": [orcid], "reviewer_id_list": [reviewer_id]}
        else:
            dataset[pmid]["orcid_list"].append(orcid)
            dataset[pmid]["reviewer_id_list"].append(reviewer_id)

    sorted_dataset = sorted(dataset.values(), key=lambda x: x["query_id"])
    json.dump(sorted_dataset, open(cfg["dataset_path"], "wt"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()