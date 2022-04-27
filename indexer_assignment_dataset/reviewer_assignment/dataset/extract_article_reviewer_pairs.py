from .config import CONFIG as cfg
import pandas as pd
import xml.etree.ElementTree as ET


def main():
    dataset = set()

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
        
        xml_path = cfg["xml_path_template"].format(pmcid)
        with open(xml_path) as read_file:
            root_node = ET.parse(read_file)
            for sub_article_node in root_node.findall("sub-article"):
                if ("article-type" in sub_article_node.attrib and sub_article_node.attrib["article-type"] == "peer-review"):
                    for contrib_group_node in sub_article_node.findall("front-stub/contrib-group"):
                        for contrib_node in contrib_group_node.findall("contrib"):
                            if ("contrib-type" in contrib_node.attrib and contrib_node.attrib["contrib-type"] == "author"):
                                surname_node = contrib_node.find("name/surname")
                                given_names_node = contrib_node.find("name/given-names")
                                contrib_id_node = contrib_node.find("contrib-id")
                                surname = ""
                                given_names = ""
                                orcid = ""
                                if surname_node is not None:
                                    surname = ET.tostring(surname_node, encoding="unicode", method="text")
                                if given_names_node is not None:
                                    given_names = ET.tostring(given_names_node, encoding="unicode", method="text")
                                if (contrib_id_node is not None and "contrib-id-type" in contrib_id_node.attrib and contrib_id_node.attrib["contrib-id-type"] == "orcid"):
                                    orcid = ET.tostring(contrib_id_node, encoding="unicode", method="text")
                                    orcid = orcid.replace("http://orcid.org/", "https://orcid.org/")
                                    orcid = orcid.replace("https://orcid.org/", "")
                                dataset.add((surname.strip(), given_names.strip(), orcid.strip(), pmid.strip(), pmcid.strip()))
    
    reviewer_assignments_list = sorted(dataset)
    df = pd.DataFrame(reviewer_assignments_list)
    df.to_csv(cfg["article_reviewer_pairs_path"], header=None, index=False)

          
if __name__ == "__main__":
    main()