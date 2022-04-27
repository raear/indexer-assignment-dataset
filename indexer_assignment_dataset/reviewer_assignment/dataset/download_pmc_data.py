from .config import CONFIG as cfg
import os
import time
from ...shared.helper import make_parent_dir
import urllib.request


def main():
    data_path_template = cfg["data_path_template"]
    ftp_url_template = cfg["ftp_url_template"]
    index_path = cfg["ftp_index_path"]
    
    data_path_ex = data_path_template.format(1)
    make_parent_dir(data_path_ex)
    
    index_line_list = open(index_path).readlines()

    count = 0
    for line in index_line_list:
        count += 1
        print(count, end="\r")
        ftp_path, journal, pmcid = line.strip().split("\t")[:3]
        save_path = data_path_template.format(pmcid)
        if os.path.isfile(save_path):
            continue
        url = ftp_url_template.format(ftp_path)
        time.sleep(1/3)
        urllib.request.urlretrieve(url, save_path)


if __name__ == "__main__":    
    main()