import gzip
import json
import pickle
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def compute_metrics(pred):
    labels = pred.label_ids
    preds = expit(pred.predictions)
    preds_argmax = preds.argmax(-1)
    labels_argmax = labels.argmax(-1)
    preds = (preds >= 0.5).astype("float")
    accuracy = accuracy_score(labels.reshape(-1), preds.reshape(-1))
    cat_accuracy = accuracy_score(labels_argmax, preds_argmax)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    return {"accuracy": accuracy, "categorical_accuracy": cat_accuracy, "f1": f1, "precision": precision, "recall": recall }


def create_dataset(ds_cfg, tokenizer, is_val, limit=1000000000):
    journal_indexer_lookup = pickle.load(open(ds_cfg["journal_indexer_lookup_path"], "rb"))
    if is_val:
        data = json.load(gzip.open(ds_cfg["val_set_path"], mode="rt", encoding="utf8"))
    else:
        data = json.load(gzip.open(ds_cfg["train_set_path"], mode="rt", encoding="utf8"))
    data = data[:limit]
    dataset = Dataset(ds_cfg, tokenizer, journal_indexer_lookup, data)
    return dataset


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, ds_cfg, tokenizer, journal_indexer_lookup, data):
        self._ds_cfg = ds_cfg
        self._tokenizer = tokenizer
        self._journal_indexer_lookup = journal_indexer_lookup
        self._data = data
        self._all_indexer_nums = set(range(1, self._ds_cfg["num_indexers"] + 1))
        
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, idx):
        example = self._data[idx]
        
        title = example["title"]
        abstract = example["abstract"]
        indexer_num = example["indexer_num"]
        nlmid = example["journal_nlmid"]
  
        inputs = self._tokenizer(title, abstract, max_length=self._ds_cfg["max_length"], padding="max_length", truncation="only_second", return_tensors="pt")
        item = {key: val[0] for key, val in inputs.items()}

        labels = self._create_labels(nlmid, indexer_num)
        item["labels"] = labels

        mask = self._create_mask(nlmid, indexer_num)
        item["labels_mask"] = mask
        return item

    def _create_labels(self, nlmid, indexer_num):
        if self._ds_cfg["equal_expertise"]:
            indexer_num_list = list(self._journal_indexer_lookup[nlmid])
        else:
            indexer_num_list = [indexer_num]
        labels = self._multi_hot(indexer_num_list)
        return labels

    def _create_mask(self, nlmid, indexer_num):
        mask_indexers = []
        if self._ds_cfg["loss_masking"]:
            other_journal_indexers = self._journal_indexer_lookup[nlmid] - set([indexer_num])
            mask_indexers.extend(other_journal_indexers)
        mask = self._multi_hot(mask_indexers)
        return mask

    def _multi_hot(self, indexer_num_list):
        indexer_nums = torch.tensor(indexer_num_list, dtype=torch.int64)
        indices = indexer_nums - 1
        vector = torch.zeros(self._ds_cfg["num_indexers"], dtype=torch.float).scatter_(0, indices, 1.)
        return vector


class PredDataset(torch.utils.data.Dataset):
    
    def __init__(self, ds_cfg, tokenizer, filtered_dataset):
        self._ds_cfg = ds_cfg
        self._tokenizer = tokenizer
        self._filtered_dataset = filtered_dataset

    def __len__(self):
        return len(self._filtered_dataset)
        
    def __getitem__(self, idx):
        example = self._filtered_dataset[idx]
        
        title = example["title"]
        abstract = example["abstract"]
  
        inputs = self._tokenizer(title, abstract, max_length=self._ds_cfg["max_length"], padding="max_length", truncation="only_second", return_tensors="pt")
        item = {key: val[0] for key, val in inputs.items()}

        return item