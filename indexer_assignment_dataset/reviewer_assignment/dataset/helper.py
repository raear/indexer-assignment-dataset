from .config import CONFIG as cfg
from ftplib import FTP
import json


def close_ftp(ftp):
    try:
        ftp.close()
    except:
        pass


def create_ftp():
    ftp = FTP(cfg["ftp_host"])  
    ftp.login()
    ftp.cwd(cfg["ftp_cwd"])
    return ftp


def create_qrels(input_dataset_path, qrels_path):
    dataset = json.load(open(input_dataset_path))

    qrels = {}
    for example in dataset:
        query_id = example["query_id"]
        if query_id not in qrels:
            qrels[query_id] = {}
        for reviewer_id in example["reviewer_id_list"]:
            qrels[query_id][reviewer_id] = 1

    with open(qrels_path, "wt") as qrels_file:
        for query_id in sorted(qrels):
            reviewer_id_list = qrels[query_id]
            for reviewer_id in sorted(reviewer_id_list):
                line = f"{query_id}\tQ0\t{reviewer_id}\t1\n"
                qrels_file.write(line)