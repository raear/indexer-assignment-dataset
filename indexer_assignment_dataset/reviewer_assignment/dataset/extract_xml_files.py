from .config import CONFIG as cfg
import os
import re
from ...shared.helper import make_parent_dir
import tarfile


def main():
    index_path = cfg["ftp_index_path"]
    pattern = cfg["xml_filename_pattern"]
    read_path_template = cfg["data_path_template"]
    write_path_template = cfg["xml_path_template"]

    write_path_ex = write_path_template.format(1)
    make_parent_dir(write_path_ex)
   
    index_line_list = open(index_path).readlines()
    for idx, line in enumerate(index_line_list):
        print(f"{idx:05}/{len(index_line_list):05}", end="\r")
        pmcid = line.strip().split("\t")[2]
        
        read_path = read_path_template.format(pmcid)
        write_path = write_path_template.format(pmcid)

        if os.path.isfile(write_path):
            continue

        retry_count = 0
        while(retry_count <= cfg["max_retrys"]):
            try:
                tar = tarfile.open(read_path, "r:gz")
                for f in tar:
                    if re.search(pattern, f.name):
                        f_contents = tar.extractfile(f.name).read()
                        open(write_path, "wb").write(f_contents)
                break
            except Exception as e:
                print(" "*15, end="\r")
                print(f"PMCID: {pmcid}, Retry count: {retry_count}.")
                print(e)
                retry_count += 1
                

if __name__ == "__main__":
    main()