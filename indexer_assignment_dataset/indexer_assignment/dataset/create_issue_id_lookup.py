from .config import CONFIG as cfg
from datetime import date
import gzip
import json
import pandas as pd
from ...shared.helper import to_date


def add_issue_key(dataset):
    for example in dataset:
        example["date_completed_date"] = to_date(example["date_completed"])
        example["issue_key"] = "_".join([str(example["indexer_id"]), str(example["journal_nlmid"]), str(example["volume"]), str(example["issue"])])


def assign_issue_ids(issue_list, start_id):
    save_data = { "pmid": [], "issue_id": [], "indexing_method": [], "count": [], "indexer_id": [], "date_completed": [], "journal_nlmid": [], "volume": [], "issue": []}

    issue_id = start_id
    for issue in issue_list:
        count = len(issue)
        for example in issue:
            save_data["pmid"].append(example["pmid"])
            save_data["issue_id"].append(issue_id)
            save_data["indexing_method"].append(example["indexing_method"])
            save_data["count"].append(count)
            save_data["indexer_id"].append(example["indexer_id"])
            save_data["date_completed"].append(example["date_completed"])
            save_data["journal_nlmid"].append(example["journal_nlmid"])
            save_data["volume"].append(example["volume"])
            save_data["issue"].append(example["issue"])
        issue_id += 1

    return save_data


def create_lookup(dataset_path, save_path, start_id):
    dataset = json.load(gzip.open(dataset_path, "rt", encoding="utf8"))

    add_issue_key(dataset)
    issue_list = create_issue_list(dataset)
    issue_list = split_issues(issue_list)
    save_data = assign_issue_ids(issue_list, start_id)

    df = pd.DataFrame(save_data)
    df.to_csv(save_path, index=False, header=True)


def create_issue_list(dataset):
    sorted_dataset = sorted(dataset, key=lambda x: (int(x["indexer_id"]), x["date_completed_date"], x["issue_key"], int(x["pmid"])))

    current_indexer_id = -1
    current_date_completed_date = date(1800, 1, 1)
    previous_day_issues = {}
    current_day_issues = {}
    issue_list = []
    for example in sorted_dataset:

        indexer_id = example["indexer_id"]
        date_completed_date = example["date_completed_date"]
        if indexer_id != current_indexer_id:
            current_indexer_id = indexer_id
            current_date_completed_date = date_completed_date
            issue_list.extend(previous_day_issues.values())
            issue_list.extend(current_day_issues.values())
            previous_day_issues = {}
            current_day_issues = {}
        
        if date_completed_date != current_date_completed_date:
            issue_list.extend(previous_day_issues.values())
            previous_day_issues = dict(current_day_issues.items())
            current_day_issues = {}
            current_date_completed_date = date_completed_date
        
        issue_key = example["issue_key"]

        if issue_key in previous_day_issues:
            current_day_issues[issue_key] = previous_day_issues.pop(issue_key)

        if issue_key not in current_day_issues:
            current_day_issues[issue_key] = []

        current_day_issues[issue_key].append(example)

    issue_list.extend(previous_day_issues.values())
    issue_list.extend(current_day_issues.values())

    return issue_list


def split_issues(issue_list):
    split_issue_list = []
    for issue in issue_list:
        indexing_method = issue[0]["indexing_method"].lower()
        threshold = cfg["standard_issue_size"] if indexing_method == "human" else cfg["mtifl_issue_size"]
        issue_length = len(issue)
        num_sub_issues = round(issue_length/threshold)
        if num_sub_issues < 2:
            split_issue_list.append(issue)
        else:
            for idx in range(num_sub_issues):
                start_idx = idx*threshold
                end_idx = start_idx + threshold
                if (idx + 1) == num_sub_issues:
                    end_idx = None
                sub_issue = issue[start_idx: end_idx]
                split_issue_list.append(sub_issue)
            
    return split_issue_list


def main():
    create_lookup(cfg["contractor_val_set_path"], cfg["contractor_val_set_issue_id_lookup_path"], start_id=int(1e7))
    create_lookup(cfg["contractor_test_set_path"], cfg["contractor_test_set_issue_id_lookup_path"], start_id=int(2e7))
    create_lookup(cfg["in_house_val_set_path"], cfg["in_house_val_set_issue_id_lookup_path"], start_id=int(3e7))
    create_lookup(cfg["in_house_test_set_path"], cfg["in_house_test_set_issue_id_lookup_path"], start_id=int(4e7))


if __name__ == "__main__":
    main()