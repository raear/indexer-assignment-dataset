from .config import CONFIG as cfg
import numpy as np
import pandas as pd
import pickle
from ...shared.helper import make_parent_dir


def error_analysis(type_, method):
    journal_ids_path = cfg["journal_nlmids_path_template"].format(type=type_, method=method)
    pred_assignments_path = cfg["pred_assignments_path_template"].format(type=type_, method=method)
    save_path = cfg["error_analysis_path"].format(type=type_, method=method)
    summary_path = cfg["error_analysis_summary_path"].format(type=type_, method=method)
    true_assignments_path = cfg["true_assignments_path_template"].format(type=type_, method=method)
    weights_path = cfg["weights_path_template"].format(type=type_, method=method)

    make_parent_dir(save_path)

    journal_ids = np.load(journal_ids_path)
    y_pred = np.load(pred_assignments_path)
    y_true = np.load(true_assignments_path)
    weights = np.load(weights_path)
    
    journal_descriptor_lookup = pickle.load(open(cfg["journal_descriptor_lookup_path"], "rb"))
    journal_indexer_lookup = pickle.load(open(cfg["journal_indexer_lookup_path"], "rb"))
    indexer_journal_descriptor_lookup = pickle.load(open(cfg["indexer_journal_descriptor_lookup_path"], "rb"))

    example_count = len(y_true)
    data = { "correct" : [0]*example_count, 
             "incorrect_expertise" : [0]*example_count, 
             "incorrect_descriptor_expertise": [0]*example_count, 
             "incorrect_no_expertise": [0]*example_count, 
             "incorrect_not_assigned": [0]*example_count,
             "notes": [""]*example_count}

    for i in range(example_count):
        weight = weights[i]

        predictions = y_pred[i]
        J = np.nonzero(predictions)[0]
        assign_count = len(J)
        if assign_count > 1:
            raise ValueError(f"More than one assignment for example number: {i + 1}")

        if assign_count == 0:
            data["incorrect_not_assigned"][i] = weight
            continue

        j = J[0]
        indexer_num = j + 1

        is_correct = bool(y_true[i,j])
        if is_correct:
            data["correct"][i] = weight
            continue

        journal_id = journal_ids[i]
        has_expertise = indexer_num in journal_indexer_lookup[journal_id]
        if has_expertise:
            data["incorrect_expertise"][i] = weight
            continue

        has_descriptor_expertise = False
        if journal_id in journal_descriptor_lookup:
            journal_desciptors = journal_descriptor_lookup[journal_id]
            indexer_journal_descriptors = indexer_journal_descriptor_lookup[indexer_num]
            diff = journal_desciptors - indexer_journal_descriptors
            if len(diff) == 0:
                has_descriptor_expertise = True
            else:
                match = journal_desciptors.intersection(indexer_journal_descriptors)
                data["notes"][i] = "Match: " + "|".join(sorted(match)) + " Diff: " + "|".join(sorted(diff))
        else:
            data["notes"][i] = "No journal descriptors."

        if has_descriptor_expertise:
            data["incorrect_descriptor_expertise"][i] = weight
            continue

        data["incorrect_no_expertise"][i] = weight

    df = pd.DataFrame(data, columns=list(data.keys()))
    df.to_csv(save_path, header=True, index=False)

    sum_weights = np.sum(weights)
    summary_data = { key: [sum(value) / sum_weights] for key, value in data.items() if key != "notes"}
    df = pd.DataFrame(summary_data, columns=list(summary_data.keys()))
    df.to_csv(summary_path, header=True, index=False)


def main():
    error_analysis("article", "baseline")
    error_analysis("issue", "baseline")
    error_analysis("article", "tfidf")
    error_analysis("issue", "tfidf")
    error_analysis("article", "specter")
    error_analysis("issue", "specter")
    error_analysis("article", "indexer_profiles_1")
    error_analysis("issue", "indexer_profiles_1")
    error_analysis("article", "indexer_profiles_2")
    error_analysis("issue", "indexer_profiles_2")
    error_analysis("article", "indexer_profiles_3")
    error_analysis("issue", "indexer_profiles_3")


if __name__ == "__main__":
    main()