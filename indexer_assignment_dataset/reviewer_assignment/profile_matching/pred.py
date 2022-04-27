from .config import CONFIG as cfg
import json
import pandas as pd
from ...shared.helper import load_run, save_run


def transform(search_results_path, predictions_path):
    documents_data = pd.read_csv(cfg["documents_path"], sep="\t", header=None, dtype={2:str, 3:str}, keep_default_na=False)
    exclude_list = json.load(open(cfg["exclude_list_path"]))
    reviewer_id_lookup = { doc_id: reviewer_id for _, (doc_id, reviewer_id, __, ___) in list(documents_data.iterrows()) }

    results = load_run(search_results_path)

    run = {}
    for q_id_int in results:
        run[q_id_int] = {}
        for doc_id_int, score in sorted(results[q_id_int].items(), key=lambda x: x[1], reverse=True)[:cfg["search_top_n"]]:
            reviewer_id = reviewer_id_lookup[int(doc_id_int)]
            if reviewer_id in exclude_list[str(q_id_int)]:
                continue
            if reviewer_id not in run[q_id_int]:
                run[q_id_int][reviewer_id] = [score]
            else:
                run[q_id_int][reviewer_id].append(score)

    run_aggregate = {}
    for q_id_int in run:
        run_aggregate[q_id_int] = {}
        for r_id in run[q_id_int]:
            sorted_scores = run[q_id_int][r_id]
            sorted_scores = sorted(sorted_scores, reverse=True)
            top_sorted_scores = sorted_scores[:cfg["pred_top_n"]]
            run_aggregate[q_id_int][r_id] = 0.
            for idx, score in enumerate(top_sorted_scores):
                denominator = idx + 1
                run_aggregate[q_id_int][r_id] += score/denominator

    save_run(predictions_path, run_aggregate)


def main():
    transform(cfg["tfidf_search_results_path"], cfg["tfidf_predictions_path"])
    transform(cfg["specter_search_results_path"], cfg["specter_predictions_path"])


if __name__ == "__main__":
    main()