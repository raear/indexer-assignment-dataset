from .config import CONFIG as cfg
import gzip
import json
import os.path
from ...shared.helper import make_parent_dir, parse_citation_node
import xml.etree.ElementTree as ET


def main():   
    start_file_num = 1
    end_file_num = cfg["medline_data_num_files"]
    
    downloaded_data_path_template = cfg["medline_data_path_template"]
    extracted_data_path_template = cfg["extracted_data_path_template"]

    extracted_data_ex = extracted_data_path_template.format(1)
    make_parent_dir(extracted_data_ex)

    for file_num in range(start_file_num, end_file_num + 1):
        print(f"Extracting citation data: {file_num:04}/{end_file_num:04}", end="\r")
        downloaded_data_path = downloaded_data_path_template.format(file_num)
        extracted_data_path = extracted_data_path_template.format(file_num)
        if os.path.isfile(extracted_data_path):
            continue
        with gzip.open(downloaded_data_path, "rt", encoding="utf8") as read_file, \
             gzip.open(extracted_data_path, "wt", encoding="utf8") as write_file:
            root_node = ET.parse(read_file)
            extracted_data = [parse_citation_node(citation_node) for citation_node in root_node.findall("PubmedArticle/MedlineCitation")]
            json.dump(extracted_data, write_file, ensure_ascii=False, indent=4)
    
    print(" "*50, end="\r")


if __name__ == "__main__":
    main()