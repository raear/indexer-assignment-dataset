from .config import CONFIG as cfg
import json
import pandas as pd
import requests
from time import sleep


def main():
    reviewer_publication_list = {}

    pairs = pd.read_csv(cfg["article_reviewer_pairs_path"], sep=",", header=None, dtype={3:str}, keep_default_na=False)
    for idx, (surname, given_names, orcid, _, __) in list(pairs.iterrows())[:]:
        print(f"{idx:04}/{pairs.shape[0]:04}", end="\r")
        
        if not orcid or pd.isna(orcid) or orcid in reviewer_publication_list:
            continue

        reviewer_publications = { "surname": surname, "given_names": given_names, "orcid": orcid, "reviewer_id": None}
        reviewer_publication_list[orcid] = reviewer_publications
 
        response_json = try_request(orcid)
        if response_json:
            publications = parse_publications(response_json)
            reviewer_publications["publications"] = publications

    json.dump(list(reviewer_publication_list.values()), open(cfg["reviewer_publications_path"], "wt"), indent=4, ensure_ascii=False)


def parse_publications(works):
    article_list = []

    for item in works["group"]:
        article = { "title": None, "doi": None, "pmid": None, "year": None, "journal": None}
        article_list.append(article)

        work_summary = item["work-summary"][0]
        title = work_summary["title"]["title"]["value"].strip() if work_summary["title"] else None
        article["title"] = title

        year = work_summary["publication-date"]["year"]["value"].strip() if work_summary["publication-date"] else None
        article["year"] = year

        journal = work_summary["journal-title"]["value"].strip() if work_summary["journal-title"] else None
        article["journal"] = journal

        for external_id in item["external-ids"]["external-id"]:
            if external_id["external-id-type"] == "pmid":
                pmid = external_id["external-id-value"]
                pmid = _correct_pmid(pmid)
                if _pmid_is_valid(pmid):
                    article["pmid"] = pmid
                else:
                    print(f"Invalid PMID: {pmid}")
            elif external_id["external-id-type"] == "doi":
                doi = external_id["external-id-value"].strip()
                doi = doi.replace("http://dx.doi.org/", "")
                doi = doi.replace("https://dx.doi.org/", "")
                doi = doi.replace("http://doi.org/", "")
                doi = doi.replace("https://doi.org/", "")
                article["doi"] = doi

    return article_list


def try_request(orcid):
    response_json = None

    query = cfg["orcid_api_url_template"].format(orcid)

    retry_count = 0
    while (retry_count <= cfg["max_retrys"]):
        try:
            r = requests.get(query, headers={"Accept": "application/vnd.orcid+json", "Authorization type and Access token": f'Bearer {cfg["orcid_api_token"]}'})
            response_json = r.json()
            sleep(0.1)
            break
        except Exception as e:
            print(" "*15, end="\r")
            print(f"ORCID: {orcid}, Retry count: {retry_count}")
            print(e)
            retry_count += 1

    return response_json


def _correct_pmid(pmid):
    pmid = pmid.replace("MEDLINE:", "")
    pmid = pmid.replace("PMID:", "")
    pmid = pmid.strip()
    return pmid


def _pmid_is_valid(pmid):
    try:
        pmid = int(pmid)
    except:
        return False

    if pmid < 1:
        return False

    if pmid > 40000000:
        return False

    return True


if __name__ == "__main__":
    main()