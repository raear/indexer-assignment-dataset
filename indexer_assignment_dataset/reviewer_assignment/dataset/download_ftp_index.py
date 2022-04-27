from .config import CONFIG as cfg
from . import helper
from io import BytesIO
from ...shared.helper import make_parent_dir


def main():
    journal_name = cfg["journal_name"]
    index_path = cfg["ftp_index_path"]
    make_parent_dir(index_path)

    with helper.create_ftp() as ftp:
        cmd = f'RETR {cfg["ftp_index_filename"]}'
        buffer = BytesIO()
        ftp.retrbinary(cmd, buffer.write)
        index_text = buffer.getvalue().decode("utf-8")

    with open(index_path, "wt") as write_file:
        line_list = index_text.split("\n")[1:]
        for line in line_list:
            if line:
                journal = line.strip().split("\t")[1]
                if journal.startswith(journal_name):   
                    write_file.write(line)
                    write_file.write("\n")


if __name__ == "__main__":    
    main()