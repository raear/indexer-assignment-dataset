import numpy as np
from sklearn.metrics import accuracy_score, ndcg_score


def accuracy(y_pred, y_true, weights):
    score = accuracy_score(y_pred, y_true, sample_weight=weights)
    return score


def get_journal_counts(journal_ids, y, weights):
    journal_counts = {}
    nonzero = np.nonzero(y)
    for i, j in zip(nonzero[0], nonzero[1]):
        journal_id = journal_ids[i]
        indexer_num = j + 1
        if journal_id not in journal_counts:
            journal_counts[journal_id] = {}
        if indexer_num not in journal_counts[journal_id]:
            journal_counts[journal_id][indexer_num] = 0
        journal_counts[journal_id][indexer_num] += weights[i]
    return journal_counts    


def ranking_metrics(y_true_counts, y_pred_counts, indexer_count, use_weights):
    journal_idx_lookup = { journal_id: idx for idx, journal_id in enumerate(sorted(y_true_counts)) }
    num_journals = len(journal_idx_lookup)

    def _create_zeros_array():
        size = (num_journals, indexer_count)
        zeros = np.zeros(size, dtype=np.int)
        return zeros

    def _add_counts(array, counts):
        for journal_id in counts:
            if journal_id in journal_idx_lookup:
                journal_idx = journal_idx_lookup[journal_id]
                for indexer_num in counts[journal_id]:
                    indexer_idx = indexer_num - 1
                    array[journal_idx][indexer_idx] = counts[journal_id][indexer_num]

    y_true_counts_array = _create_zeros_array()
    _add_counts(y_true_counts_array, y_true_counts)

    y_pred_counts_array = _create_zeros_array()
    _add_counts(y_pred_counts_array, y_pred_counts)

    if use_weights:
        journal_counts = np.sum(y_true_counts_array, axis=1)
        total_count = np.sum(y_true_counts_array)
        sample_weight = journal_counts / total_count
    else:
        sample_weight = np.ones(num_journals)
        
    ndcg = ndcg_score(y_true_counts_array, y_pred_counts_array, sample_weight=sample_weight)

    return ndcg