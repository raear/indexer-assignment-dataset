from .config import CONFIG as cfg
import os.path
from ..shared.helper import enumerate_year_week_numbers
from ...shared.helper import load_run, save_run


def aggregate_func(query_search_results):
    sorted_scores = [float(score) for score in sorted(query_search_results.values(), reverse=True)]
    aggregate_score = 0.
    for idx in range(cfg["pred_top_n"]):        
        score = sorted_scores[idx]
        if cfg["pred_use_weights"]:
            score = score / (idx+1.)
        aggregate_score = aggregate_score + score
    return aggregate_score


def main():
    pred(cfg["tfidf_search_results_path_template"], cfg["tfidf_predictions_path_template"])
    pred(cfg["specter_search_results_path_template"], cfg["specter_predictions_path_template"])


def pred(search_results_path_template, predictions_path_template):
    for year, week_number in enumerate_year_week_numbers(cfg["eval_set_start_date"], cfg["eval_set_end_date"]):
        search_results_path_ex = search_results_path_template.format(year=year, week_number=week_number, indexer_num=1)
        if not os.path.isfile(search_results_path_ex):
            continue
        qrels = {}
        for indexer_num in range(1, cfg["num_indexers"] + 1):
            search_results_path = search_results_path_template.format(year=year, week_number=week_number, indexer_num=indexer_num)
            search_results = load_run(search_results_path)
            for query_id_int in search_results:
                query_search_results = search_results[query_id_int]
                aggregate_score = aggregate_func(query_search_results)
                query_id = int(query_id_int)
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][indexer_num] = aggregate_score
        
        predictions_path = predictions_path_template.format(year=year, week_number=week_number)
        save_run(predictions_path, qrels)


if __name__ == "__main__":
    main()