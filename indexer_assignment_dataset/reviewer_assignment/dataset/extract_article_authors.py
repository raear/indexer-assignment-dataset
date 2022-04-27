from .config import CONFIG as cfg
import json
import xml.etree.ElementTree as ET


def main():
    article_authors = {}

    index_line_list = open(cfg["ftp_index_path"]).readlines()
    for idx, line in enumerate(index_line_list):
        print(f"{idx:05}/{len(index_line_list):05}", end="\r")
        
        line_data = line.strip().split("\t")
        pmcid = line_data[2].strip()
        pmid = line_data[3].strip()
        pmid = pmid.replace("PMID:", "")
        if not pmid:
            continue
        pmid = str(int(pmid))
        article_authors[pmid] = []
        
        xml_path = cfg["xml_path_template"].format(pmcid)
        with open(xml_path) as read_file:
            root_node = ET.parse(read_file)
            for contrib_node in root_node.findall("front/article-meta/contrib-group/contrib"):
                if ("contrib-type" in contrib_node.attrib and contrib_node.attrib["contrib-type"] == "author"):
                    surname_node = contrib_node.find("name/surname")
                    given_names_node = contrib_node.find("name/given-names")
                    contrib_id_node = contrib_node.find("contrib-id")
                    surname = None
                    given_names = None
                    orcid = None
                    if surname_node is not None:
                        surname = ET.tostring(surname_node, encoding="unicode", method="text")
                    if given_names_node is not None:
                        given_names = ET.tostring(given_names_node, encoding="unicode", method="text")
                    if (contrib_id_node is not None and "contrib-id-type" in contrib_id_node.attrib and contrib_id_node.attrib["contrib-id-type"] == "orcid"):
                        orcid = ET.tostring(contrib_id_node, encoding="unicode", method="text")
                        orcid = orcid.replace("http://orcid.org/", "https://orcid.org/")
                        orcid = orcid.replace("https://orcid.org/", "")
                    article_authors[pmid].append({ "surname": surname, "given_names": given_names, "orcid": orcid})
    
    json.dump(article_authors, open(cfg["article_authors_path"], "wt"), indent=4, ensure_ascii=False)

          
if __name__ == "__main__":
    main()