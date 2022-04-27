from .config import CONFIG as cfg
import json
import requests
from ...shared.helper import parse_citation_node
import re
from time import sleep
import xml.etree.ElementTree as ET


def contruct_author_name(author_info):
    surname = author_info["surname"]
    given_names = author_info["given_names"]
    initials = "".join([name[0].upper() for name in given_names.split()])
    author_name = f"{surname} {initials}"
    return author_name


def main():
    reviewer_publication_list = json.load(open(cfg["reviewer_publications_path"]))
    reviewer_publications_with_metadata_list = []
    doc_id = 0
    reviewer_id = 0
    for idx, reviewer_publications in enumerate(reviewer_publication_list[:]):
        print(f"{idx:04}/{len(reviewer_publication_list):04}", end="\r")

        author = contruct_author_name(reviewer_publications)
        articles_with_metadata = {}
        for article in reviewer_publications["publications"]:

            doi = article["doi"]
            pmid = article["pmid"]

            if not pmid:                
                pmid = query_pmid(article, author)

            if not pmid:
                continue

            metadata = query_metadata(pmid)
            if not metadata:
                continue

            orcid_title = normalize_text(article["title"])
            pubone_title = normalize_text(metadata["title"])
            pubone_vernacular_title = normalize_text(metadata["vernacular_title"])
            pubone_title_and_pubone_vernacular_title = metadata["title"] + " " + metadata["vernacular_title"]
            pubone_title_and_pubone_vernacular_title = normalize_text(pubone_title_and_pubone_vernacular_title)

            if (doi or 
                orcid_title == pubone_title or 
                orcid_title == pubone_vernacular_title or 
                orcid_title == pubone_title_and_pubone_vernacular_title):
                if pmid in articles_with_metadata:
                    print(f"Duplicate pmid: {pmid}")
                    continue
                articles_with_metadata[pmid] = article
                doc_id += 1
                article["doc_id"] = doc_id
                article["pmid"] = pmid
                article["title"] = metadata["title"]
                article["abstract"] = metadata["abstract"]
                article["medline_ta"] = metadata["medline_ta"]
                article["nlmid"] = metadata["journal_nlmid"]
                article["pub_date"] = metadata["pub_date"]
            else:
                print(f"Title mismatch")
                print(orcid_title)
                print(pubone_title)
                print(pubone_vernacular_title)

        articles_with_metadata_list = list(articles_with_metadata.values())
        if len(articles_with_metadata_list) > 0:
            reviewer_id += 1
            reviewer_publications["reviewer_id"] = reviewer_id
            reviewer_publications["publications"] = articles_with_metadata_list
            reviewer_publications_with_metadata_list.append(reviewer_publications)
         
    json.dump(reviewer_publications_with_metadata_list, open(cfg["reviewer_publications_with_metadata_path"], "wt"), indent=4, ensure_ascii=False)


def normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]"," ",text)
    text = re.sub(r"\s+"," ",text)
    text = text.strip()
    return text


def query_metadata(pmid):
    metadata = None

    xml = None
    retry_count = 0
    while (retry_count <= cfg["max_retrys"]):
        try:
            url = cfg["efetch_url_template"].format(pmid)
            xml = requests.get(url, timeout=10).content
            sleep(1/3)
            if xml:
                break
            else:
                print(f"{pmid}: PubOne empty response, retry count: {retry_count}")
                retry_count += 1
                continue
        except Exception as e:
            print(f"{pmid}: PubOne exception, retry count: {retry_count}")
            print(e)
            retry_count += 1

    if not xml:
        print(f"PubOne empty response after {MAX_RETRYS} retrys.")
        return metadata

    try:
        root_node = ET.fromstring(xml)
    except Exception as e:
        print(f"{pmid}: Error parsing XML.")
        print(e)
        return metadata

    if root_node.tag == "PubmedBookArticleSet":
        print(f"{pmid}: Book (Ignored).")
        return metadata

    medline_citation_node = root_node.find("PubmedArticle/MedlineCitation")
    if medline_citation_node is None:
        print(f"{pmid}: No MedlineCitation node in XML.")
        return metadata
    
    metadata = parse_citation_node(medline_citation_node)
    
    return metadata


def query_pmid(article, author):
    pmid = None

    doi = article["doi"]
    if doi:
        term = f'"{doi}"[AID]'
    else:
        title = article["title"].strip()
        term = f'("{author}"[Author]) AND ("{title}"[Title])'
    query = cfg["esearch_url_template"].format(term)

    retry_count = 0
    id_list = None
    json_response = None
    while (retry_count <= cfg["max_retrys"]):
        try:
            r = requests.get(query, timeout=10)
            json_response = r.json()
            sleep(1/3)
            id_list = json_response["esearchresult"]["idlist"]
            break
        except:
            print(term)
            print(json_response)
            print(f"Retry count: {retry_count}")
            retry_count += 1
            
    if id_list:
        pmid = str(int(id_list[0]))

    return pmid


if __name__ == "__main__":
    main()