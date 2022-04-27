from .config import CONFIG as cfg
import os.path
from ...shared.helper import make_parent_dir
import urllib.request


def main():
    start_file_num = 1
    end_file_num = cfg["medline_data_num_files"]
    
    save_path_template = cfg["medline_data_path_template"]
    url_template = cfg["medline_data_url_template"]
    
    save_path_ex = save_path_template.format(1)
    make_parent_dir(save_path_ex)
    
    for file_num in range(start_file_num, end_file_num + 1):
        print(f"Downloading medline data: {file_num:04}/{end_file_num:04}", end="\r")
        url = url_template.format(file_num)
        save_path = save_path_template.format(file_num)
        if os.path.isfile(save_path):
            continue
        urllib.request.urlretrieve(url, save_path)
        
    print(" "*50, end="\r")


if __name__ == "__main__":
    main()