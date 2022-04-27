from . import assign_helper
from .config import CONFIG as cfg
from . import eval_helper
import json
from multiprocessing import Pool
import numpy as np
from ..shared.helper import enumerate_year_week_numbers
from ...shared.helper import make_parent_dir, to_date


def multiprocess_assign(_type, method, article_dataset_path, issue_id_lookup_path, predictions_path_template, add_noise, assign_issues):
    num_pools = cfg["num_pools"]

    config_path =           cfg["config_path_template"].format(type=_type, method=method)
    ids_path =              cfg["ids_path_template"].format(type=_type, method=method)
    journal_nlmids_path =   cfg["journal_nlmids_path_template"].format(type=_type, method=method)
    metrics_path =          cfg["metrics_path_template"].format(type=_type, method=method)
    pred_assignments_path = cfg["pred_assignments_path_template"].format(type=_type, method=method)
    true_assignments_path = cfg["true_assignments_path_template"].format(type=_type, method=method)
    weights_path =          cfg["weights_path_template"].format(type=_type, method=method)

    eval_set_start_date = to_date(cfg["eval_set_start_date"])
    eval_set_end_date = to_date(cfg["eval_set_end_date"])

    make_parent_dir(config_path)
    json.dump(cfg, open(config_path, "wt"), ensure_ascii=False, indent=4)

    args_list = [{ "article_dataset_path": article_dataset_path, 
                   "issue_id_lookup_path": issue_id_lookup_path, 
                   "predictions_path": predictions_path_template.format(year=year, week_number=week_number), 
                   "add_noise": add_noise, 
                   "assign_issues": assign_issues } for year, week_number in enumerate_year_week_numbers(eval_set_start_date, eval_set_end_date)]
    
    result = []
    for args in args_list:
        return_value = assign_helper.run(args)
        result.append(return_value)

    # with Pool(num_pools) as p:
    #     result = p.map(assign_helper.run, args_list)

    B_list = []
    y_list = []
    journal_nlmids_list = []
    weights_list = []
    ids_list = []
    for return_value in result:
        if return_value[0] is not None:
            B_list.append(return_value[0])
            y_list.append(return_value[1])
            journal_nlmids_list.append(return_value[2])
            weights_list.append(return_value[3])
            ids_list.append(return_value[4])

    #B, y, journal_ids, weights, pmids
    
    B = np.concatenate(B_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    journal_nlmids = np.concatenate(journal_nlmids_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)

    np.save(pred_assignments_path, B)
    np.save(true_assignments_path, y)
    np.save(journal_nlmids_path, journal_nlmids)
    np.save(weights_path, weights)
    np.save(ids_path, ids)

    accuracy = eval_helper.accuracy(B, y, weights)
    B_counts = eval_helper.get_journal_counts(journal_nlmids, B, weights)
    y_counts = eval_helper.get_journal_counts(journal_nlmids, y, weights)
    indexer_count = y.shape[1]
    ndcg  = eval_helper.ranking_metrics(y_counts, B_counts, indexer_count=indexer_count, use_weights=False)
    ndcg_w = eval_helper.ranking_metrics(y_counts, B_counts, indexer_count=indexer_count, use_weights=True)
    
    metrics_text = f"Accuracy: {accuracy:.4f}, NDCG: {ndcg:.4f}, NDCG_W: {ndcg_w:.4f}"
    print(metrics_text)

    with open(metrics_path, "wt") as _file:
        _file.write(metrics_text + "\n")


def main():    
    multiprocess_assign("article", "baseline",           cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["baseline_predictions_path_template"],           add_noise=True, assign_issues=False)
    multiprocess_assign("issue",   "baseline",           cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["baseline_predictions_path_template"],           add_noise=True, assign_issues=True)
    multiprocess_assign("article", "tfidf",              cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["tfidf_predictions_path_template"],              add_noise=True, assign_issues=False)
    multiprocess_assign("issue",   "tfidf",              cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["tfidf_predictions_path_template"],              add_noise=True, assign_issues=True)
    multiprocess_assign("article", "specter",            cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["specter_predictions_path_template"],            add_noise=True, assign_issues=False)
    multiprocess_assign("issue",   "specter",            cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["specter_predictions_path_template"],            add_noise=True, assign_issues=True)
    multiprocess_assign("article", "indexer_profiles_1", cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["indexer_profiles_1_predictions_path_template"], add_noise=True, assign_issues=False)
    multiprocess_assign("issue",   "indexer_profiles_1", cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["indexer_profiles_1_predictions_path_template"], add_noise=True, assign_issues=True)
    multiprocess_assign("article", "indexer_profiles_2", cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["indexer_profiles_2_predictions_path_template"], add_noise=True, assign_issues=False)
    multiprocess_assign("issue",   "indexer_profiles_2", cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["indexer_profiles_2_predictions_path_template"], add_noise=True, assign_issues=True)
    multiprocess_assign("article", "indexer_profiles_3", cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["indexer_profiles_3_predictions_path_template"], add_noise=True, assign_issues=False)
    multiprocess_assign("issue",   "indexer_profiles_3", cfg["article_dataset_path"], cfg["issue_id_lookup_path"], cfg["indexer_profiles_3_predictions_path_template"], add_noise=True, assign_issues=True)

      
if __name__ == "__main__":
    main()