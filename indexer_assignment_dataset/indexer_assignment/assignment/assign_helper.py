from .config import CONFIG as cfg
import cvxpy as cp
import gzip
import json
import numpy as np
import os.path
import random
import scipy.sparse
import pandas as pd
from ...shared.helper import load_run


def run(args):
    return assign(**args)


def assign(article_dataset_path, issue_id_lookup_path, predictions_path, add_noise, assign_issues):

    B, y, journal_ids, weights, ids = None, None, None, None, None
    if not os.path.isfile(predictions_path):
        return B, y, journal_ids, weights, ids

    num_indexers = cfg["num_indexers"]

    article_dataset = json.load(gzip.open(article_dataset_path, "rt", encoding="utf8"))
    random.shuffle(article_dataset)
    
    if assign_issues:
        ids, journal_ids, weights, assignments, predictions = load_issue_dataset(num_indexers, article_dataset, predictions_path, issue_id_lookup_path)
    else:
        ids, journal_ids, weights, assignments, predictions = load_dataset(article_dataset, predictions_path)

    example_count = len(ids)
    print(f"Example count: {example_count}")

    if example_count > 0:
    
        A = create_affinity_matrix(example_count, num_indexers, ids, predictions)
        
        n_p, n_i, a, edge_indices, edge_count = create_edge_adjacency_matrices(example_count, num_indexers, A, weights, add_noise)
    
        quotas = create_quota_matrix(num_indexers, weights, assignments)
        I = scipy.sparse.identity(edge_count)

        indexed_once = np.ones([example_count, 1], dtype=np.int)
        x_max = np.ones([edge_count, 1], dtype=np.float)
        x_min = np.zeros([edge_count, 1], dtype=np.float)
    
        x = cp.Variable(shape=a.shape)
        prob = cp.Problem(cp.Maximize(a.T@x),
                        [n_p @ x <= indexed_once,
                        n_i @ x <= quotas,
                        I @ x <= x_max, 
                        -I @ x <= x_min])
        prob.solve(verbose=True, max_iters=1000)
        
        B = create_pred_assignment_matrix(example_count, num_indexers, edge_indices, x.value)
        y = create_true_assignment_matrix(example_count, num_indexers, assignments)

    return B, y, np.array(journal_ids), np.array(weights), np.array(ids)


def create_affinity_matrix(example_count, indexer_count, ids, predictions):
    A = np.zeros([example_count, indexer_count], dtype=np.float)
    for example_idx, id in enumerate(ids):
        for indexer_num in range(1, indexer_count + 1):
            affinity = predictions[int(id)][int(indexer_num)]
            indexer_idx = indexer_num - 1
            A[example_idx][indexer_idx] = affinity
    return A


def create_edge_adjacency_matrices(example_count, indexer_count, A, weights, add_noise):
    edge_indices = np.nonzero(A)
    edge_count = len(edge_indices[0])
    np_data, np_i, np_j = [], [], []
    ni_data, ni_i, ni_j = [], [], []
    a = np.zeros([edge_count, 1], dtype=np.float)
    for idx in range(edge_count):
        example_idx, indexer_idx = edge_indices[0][idx], edge_indices[1][idx]
        np_data.append(1)
        np_i.append(example_idx)
        np_j.append(idx)
        ni_data.append(weights[example_idx])
        ni_i.append(indexer_idx)
        ni_j.append(idx)
        a[idx] = A[example_idx, indexer_idx]
    n_p = scipy.sparse.coo_matrix((np_data, (np_i, np_j)), shape=[example_count, edge_count])
    n_i = scipy.sparse.coo_matrix((ni_data, (ni_i, ni_j)), shape=[indexer_count, edge_count])
    if add_noise:
        rho = 1. / (example_count + 1)
        a += rho*np.random.random_sample(a.shape)
    return n_p, n_i, a, edge_indices, edge_count


def create_pred_assignment_matrix(example_count, indexer_count, edge_indices, x):
    x = np.squeeze(x)
    B = np.zeros([example_count, indexer_count], dtype=np.int)
    x = (x > 0.5).astype(np.int)
    B[edge_indices[0], edge_indices[1]] = x
    return B


def create_quota_matrix(indexer_count, weights, assignments):
    quotas = np.zeros([indexer_count, 1], dtype=np.int)
    for count, indexer_num in zip(weights, assignments):
        indexer_idx = indexer_num - 1
        quotas[indexer_idx] += count
    return quotas


def create_true_assignment_matrix(example_count, indexer_count, assignments):
    y_true = np.zeros([example_count, indexer_count], dtype=np.int)
    for example_idx, indexer_num in enumerate(assignments):
        indexer_idx = indexer_num - 1
        y_true[example_idx][indexer_idx] = 1
    return y_true


def load_dataset(article_dataset, predictions_path):
    predictions = load_run(predictions_path)

    id_list = []
    journal_id_list = []
    weight_list = []
    indexer_num_list = []
    for example in article_dataset:
        id = example["pmid"]
        if int(id) in predictions and example["indexing_method"].lower().strip() == "human":
            id_list.append(id)
            journal_id_list.append(example["journal_nlmid"])
            weight_list.append(1)
            indexer_num_list.append(example["indexer_num"])

    return id_list, journal_id_list, weight_list, indexer_num_list, predictions


def load_issue_dataset(indexer_count, article_dataset, predictions_path, issue_id_lookup_path):

    predictions = load_run(predictions_path)

    issue_data = pd.read_csv(issue_id_lookup_path, index_col=None, header=0)
    issue_id_lookup = dict(zip(issue_data.iloc[:,0], issue_data.iloc[:,1]))

    issue_lookup = {}
    for example in article_dataset:
        pmid = example["pmid"]
        if int(pmid) in predictions and example["indexing_method"].lower().strip() == "human":
            issue_id = issue_id_lookup[pmid]
            if issue_id not in issue_lookup:
                issue_lookup[issue_id] = { "id": issue_id, "journal_nlmid": example["journal_nlmid"], "indexer_num": example["indexer_num"], "pmid_list": [pmid] }
            else:
                issue_lookup[issue_id]["pmid_list"].append(pmid)

    id_list = []
    journal_id_list = []
    weight_list = []
    indexer_num_list = []
    issue_predictions = {}
    for issue in issue_lookup.values():
        id = issue["id"]
        pmid_list = issue["pmid_list"]
        count = len(pmid_list)

        id_list.append(id)
        journal_id_list.append(issue["journal_nlmid"])
        weight_list.append(count)
        indexer_num_list.append(issue["indexer_num"])

        issue_predictions[int(id)] = {}
        for pmid in pmid_list:
            for indexer_num in range(1, indexer_count + 1):
                if int(indexer_num) not in issue_predictions[int(id)]:
                    issue_predictions[int(id)][int(indexer_num)] = 0.
                article_pred = predictions[int(pmid)][int(indexer_num)]
                article_pred = article_pred / count
                issue_predictions[int(id)][int(indexer_num)] += article_pred
            
    return id_list, journal_id_list, weight_list, indexer_num_list, issue_predictions