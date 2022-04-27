from .config import CONFIG as cfg
import json
import requests
from ...shared.helper import parse_citation_node
from time import sleep
import xml.etree.ElementTree as ET


def main():
    dataset = json.load(open(cfg["dataset_path"]))
    for idx, article in enumerate(dataset[:]):
        print(f"{idx:04}/{len(dataset):04}", end="\r")
        pmid = article["pmid"]
        metadata = query_metadata(pmid)
        if metadata:
            article["title"] = metadata["title"]
            article["abstract"] = metadata["abstract"]
            article["pub_date"] = metadata["pub_date"]

    json.dump(dataset, open(cfg["dataset_with_metadata_path"], "wt"), indent=4, ensure_ascii=False)


def query_metadata(pmid):
    data = None

    retry_count = 0
    url = cfg["efetch_url_template"].format(pmid)
    while (retry_count <= cfg["max_retrys"]):
        try:
            xml = requests.get(url).content
            sleep(1/3)
            root_node = ET.fromstring(xml)
            medline_citation_node = root_node.find("PubmedArticle/MedlineCitation")
            data = parse_citation_node(medline_citation_node)
            break
        except Exception as e:
            print(f"PMID: {pmid}, Retry count: {retry_count}.")
            print(e)
            retry_count += 1
    
    return data


if __name__ == "__main__":
    main()