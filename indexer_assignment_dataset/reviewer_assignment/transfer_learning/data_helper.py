import gzip
import json
import pickle
import random
from ...shared.helper import to_date
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = preds.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall }


def create_dataset(ds_cfg, tokenizer, is_val):
    journal_indexer_lookup = pickle.load(open(ds_cfg["journal_indexer_lookup_path"], "rb"))
    train_set_max_date = to_date(ds_cfg["train_set_max_date"])
    if is_val:
        dataset = Dataset(ds_cfg, tokenizer, journal_indexer_lookup, min_date=train_set_max_date, limit=ds_cfg["val_limit"])
    else:
        dataset = Dataset(ds_cfg, tokenizer, journal_indexer_lookup, max_date=train_set_max_date)
    return dataset


def pub_date_to_pub_year(pub_date):
    if not pub_date:
        return -1
    year = int(pub_date[:4])
    return year


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, ds_cfg, tokenizer, journal_indexer_lookup, min_date=None, max_date=None, limit=None):
        self._ds_cfg = ds_cfg
        self._tokenizer = tokenizer
        self._journal_indexer_lookup = journal_indexer_lookup
        
        train_dataset = json.load(gzip.open(self._ds_cfg["train_set_path"], mode="rt", encoding="utf8"))
        val_dataset =   json.load(gzip.open(self._ds_cfg["val_set_path"]  , mode="rt", encoding="utf8"))
        dataset = train_dataset + val_dataset
        random.shuffle(dataset)

        self._dataset = dataset
        self._create_buckets()
        self._create_mapping(min_date, max_date, limit)

    def __len__(self):
        return len(self._mapping)*2
        
    def __getitem__(self, getitem_index):
        idx, mod = divmod(getitem_index, 2)

        dataset_idx = self._mapping[idx]
        example = self._dataset[dataset_idx]
        
        date_completed_str = example["date_completed"]
        date_completed = to_date(date_completed_str)
        indexer_num = example["indexer_num"]
        nlmid = example["journal_nlmid"]

        bucket_idx = self._get_bucket_index(date_completed)
        
        if mod == 0:
            label = 1
            candidate_indexer_num = indexer_num
        else:
            label = 0
            candidate_indexer_num = self._get_negative_indexer_num(nlmid, indexer_num, bucket_idx)

        history_text = self._get_history_text(candidate_indexer_num, bucket_idx, dataset_idx)
        candidate_text = self._get_text(example)
            
        inputs = self._tokenizer(history_text, candidate_text, max_length=self._ds_cfg["max_length"], padding="max_length", truncation="only_first", return_tensors="pt")
        item = {key: val[0] for key, val in inputs.items()}
        item["labels"] = torch.tensor([label])
        return item
 
    def _create_buckets(self):     
        num_buckets = (self._ds_cfg["max_year"] - self._ds_cfg["min_year"] + 1)*12
        max_bucket_idx = num_buckets - 1

        monthly_buckets = [ [set() for __ in range(self._ds_cfg["num_indexers"])] for _ in range(num_buckets)]
        min_indexer_bucket_idx = { indexer_num: max_bucket_idx for indexer_num in range(1, self._ds_cfg["num_indexers"] + 1) }

        for example_idx, example in enumerate(self._dataset):
            date_completed_str = example["date_completed"]
            date_completed_date = to_date(date_completed_str)
            bucket_idx = self._get_bucket_index(date_completed_date)
            indexer_num = example["indexer_num"]
            indexer_idx = indexer_num - 1
            monthly_buckets[bucket_idx][indexer_idx].add(example_idx)
            if bucket_idx < min_indexer_bucket_idx[indexer_num]:
                min_indexer_bucket_idx[indexer_num] = bucket_idx

        history_buckets = [{} for _ in range(num_buckets)]
        for indexer_idx in range(self._ds_cfg["num_indexers"]):
            indexer_num = indexer_idx + 1
            start_bucket_idx = min_indexer_bucket_idx[indexer_num] + self._ds_cfg["history_duration"]
            for end_idx in range(start_bucket_idx, num_buckets):
                start_idx = end_idx - self._ds_cfg["history_duration"]
                history = set()
                for bucket_idx in range(start_idx, end_idx + 1):
                    history.update(monthly_buckets[bucket_idx][indexer_idx])
                if len(history) > 0:
                    history_buckets[end_idx][indexer_num] = history

        self._buckets = history_buckets

    def _create_mapping(self, min_date, max_date, limit):
        mapping = []
        for idx, example in enumerate(self._dataset):
            date_completed_str = example["date_completed"]
            indexer_num = example["indexer_num"]
            nlmid = example["journal_nlmid"]

            date_completed = to_date(date_completed_str)

            if min_date and date_completed < min_date:
                continue

            if max_date and date_completed > max_date:
                continue

            bucket_idx = self._get_bucket_index(date_completed)
            if indexer_num not in self._buckets[bucket_idx]:
                continue

            negative_indexer_num = self._get_negative_indexer_num(nlmid, indexer_num, bucket_idx)
            if negative_indexer_num < 0:
                continue
            mapping.append(idx)

        if limit:
            mapping = mapping[:limit]
            
        self._mapping = mapping

    def _get_bucket_index(self, date):
        bucket_idx = (date.year - self._ds_cfg["min_year"])*12 + (date.month - 1)
        return bucket_idx

    def _get_history_text(self, indexer_num, bucket_idx, dataset_idx):
        history = self._buckets[bucket_idx][indexer_num]
        history.discard(dataset_idx)
        sampled_history = self._sample_history(history)
        history_text  = "|".join([self._get_text(self._dataset[idx]) for idx in sampled_history if idx != dataset_idx])
        return history_text

    def _get_negative_indexer_num(self, nlmid, indexer_num, bucket_idx):
        negative_candidates = set(self._buckets[bucket_idx])
        negative_candidates.discard(indexer_num)
        journal_indexers_nums = self._journal_indexer_lookup[nlmid]
        negative_candidates = negative_candidates - journal_indexers_nums
        negative_candidates = list(negative_candidates)
        if len(negative_candidates) == 0:
            return -1
        negative_indexer_num = random.sample(negative_candidates, k=1)[0]
        return negative_indexer_num

    def _get_text(self, example):
        text = example["title"]
        if self._ds_cfg["include_abstracts"]:
            text += " "
            text += example["abstract"]
        return text

    def _sample_history(self, history):
        history = list(history)
        history_count = len(history)
        k = min(history_count, self._ds_cfg["history_length"])
        sampled_history = random.sample(history, k=k)
        return sampled_history


class ReviewerAssignmentDatasetBase(torch.utils.data.Dataset):

    def __init__(self, ds_cfg, tokenizer, order_by_pub_year):
        self._ds_cfg = ds_cfg
        self._tokenizer = tokenizer
        self._order_by_pub_year = order_by_pub_year
    
    def _get_text(self, article_info):
        text = article_info["title"]
        if self._ds_cfg["include_abstracts"]:
            text += " "
            text += article_info["abstract"]
        return text

    def _get_reviewer_text(self, publication_list):
        history_length = self._ds_cfg["history_length"]
        ordered_publications = None
        if self._order_by_pub_year:
            ordered_publications = sorted(publication_list, key=lambda x: pub_date_to_pub_year(x["pub_date"]), reverse=True)[:history_length]
        else:
            publication_count = len(publication_list)
            k = min(publication_count, history_length)
            ordered_publications = list(random.sample(publication_list, k=k))
        reviewer_text = "|".join([self._get_text(article_info) for article_info in ordered_publications])
        return reviewer_text

    def _get_inputs(self, query, reviewer):
        query_text = self._get_text(query)

        reviewer_publications = reviewer["publications"]
        reviewer_text = self._get_reviewer_text(reviewer_publications)
        inputs = self._tokenizer(reviewer_text, query_text, max_length=self._ds_cfg["max_length"], padding="max_length", truncation="only_first", return_tensors="pt")
        item = {key: val[0] for key, val in inputs.items()}
        return item


class ReviewerAssignmentDataset(ReviewerAssignmentDatasetBase):
    
    def __init__(self, ds_cfg, tokenizer, order_by_pub_year, qrels, queries, reviewers, exclude_list):
        super().__init__(ds_cfg, tokenizer, order_by_pub_year)
        self._qrels = qrels
        
        self._qrel_list = [(int(q_id), int(r_id)) for q_id in qrels for r_id in qrels[q_id]]
        self._query_lookup =    {int(query["query_id"]): query for query in queries}
        self._reviewer_lookup = {int(reviewer["reviewer_id"]): reviewer for reviewer in reviewers}
        self._reviewer_ids = set(self._reviewer_lookup)
        self._exclude_list = exclude_list
        
        self._length = 2*len(self._qrel_list)

    def __len__(self):
        return self._length
        
    def __getitem__(self, idx):
        qrel_idx, is_negative = divmod(idx, 2)

        q_id, positive_r_id = self._qrel_list[qrel_idx]

        query = self._query_lookup[q_id]

        if is_negative:
            label = 0
            query_postives = set([int(r_id) for r_id in self._qrels[q_id]])
            author_reviewer_ids = set([int(r_id) for r_id in self._exclude_list[str(q_id)]])
            candidate_negatives = self._reviewer_ids - query_postives - author_reviewer_ids
            candidate_negatives = list(candidate_negatives)
            negative_r_id = random.choice(candidate_negatives)
            reviewer = self._reviewer_lookup[negative_r_id]
        else:
            label = 1
            reviewer = self._reviewer_lookup[positive_r_id]

        item = self._get_inputs(query, reviewer)
        item["labels"] = torch.tensor([label])
        return item


class ReviewerAssignmentPredDataset(ReviewerAssignmentDatasetBase):
    
    def __init__(self, ds_cfg, tokenizer, order_by_pub_year, queries, reviewers):
        super().__init__(ds_cfg, tokenizer, order_by_pub_year)
        self._queries = queries
        self._reviewers = reviewers
        
        self._query_count = len(self._queries)
        self._reviewer_count = len(self._reviewers)
        self._length = self._query_count * self._reviewer_count

    def __len__(self):
        return self._length
        
    def __getitem__(self, idx):
        query_idx, reviewer_idx = divmod(idx, self._reviewer_count)
        
        query = self._queries[query_idx]
        reviewer = self._reviewers[reviewer_idx]
        
        item = self._get_inputs(query, reviewer)
        return item