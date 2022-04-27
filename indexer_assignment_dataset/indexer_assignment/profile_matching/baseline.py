from .config import CONFIG as cfg
import gzip
import json
import pickle
from ..shared.helper import enumerate_year_week_numbers, filter_dataset_by_weeknumber
from ...shared.helper import make_parent_dir, save_run 


def main():
    eval_dataset = json.load(gzip.open(cfg["eval_dataset_path"], mode="rt", encoding="utf8"))
    journal_indexer_lookup = pickle.load(open(cfg["eval_journal_indexer_lookup_path"], "rb"))

    baseline_predictions_path_ex = cfg["baseline_predictions_path_template"].format(year=2018, week_number=1)
    make_parent_dir(baseline_predictions_path_ex)
   
    for year, week_number in enumerate_year_week_numbers(cfg["eval_set_start_date"], cfg["eval_set_end_date"]):
        
        filtered_dataset = filter_dataset_by_weeknumber(eval_dataset, year, week_number)
        if len(filtered_dataset) == 0:
            continue
        
        run = {}
        for example in filtered_dataset:
            q_id = example["pmid"]
            nlmid = example["journal_nlmid"]
            run[q_id] = {}
            for indexer_num in range(1, cfg["num_indexers"] + 1):
                score = 0.
                if (nlmid in journal_indexer_lookup and 
                   indexer_num in journal_indexer_lookup[nlmid]):
                    score = 1.
                run[q_id][indexer_num] = score
    
        results_path = cfg["baseline_predictions_path_template"].format(year=year, week_number=week_number)
        save_run(results_path, run)


if __name__ == "__main__":
    main()