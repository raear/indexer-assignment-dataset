import argparse
import json
import os
import time
from transformers import TrainingArguments


_base_dir = "/slurm_storage/raear"
_indexer_dataset_dir = f"{_base_dir}/input-data/indexer-assignment-dataset/indexer-assignment"
_reviewer_dataset_dir = f"{_base_dir}/input-data/indexer-assignment-dataset/reviewer-assignment"

SLURM_JOB_ID_ENV_VAR = "SLURM_JOB_ID"

CONFIG = {
    "cache_dir": f"{_base_dir}/cache/huggingface/transformers",
    "indexer_assignment_dataset": {
        "history_duration": 12,
        "history_length": 50,
        "include_abstracts": False,
        "journal_indexer_lookup_path": f"{_indexer_dataset_dir}/Contractor_Journal_Indexer_Lookup.pkl",
        "max_length": 512,
        "max_year": 2019,
        "min_year": 2011,
        "num_indexers": 102,
        "train_set_max_date": "2017-12-31",
        "train_set_path": f"{_indexer_dataset_dir}/Contractor_Train_Set.json.gz",
        "val_limit": 40000,
        "val_set_path": f"{_indexer_dataset_dir}/Contractor_Val_Set.json.gz",
    },
    "model": {
        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "tokenizer_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    },
    "pred":
    {
        "config_filename": "pred_config.json",
        "limit": 1000000000,
        "results_filename": "test_predictions.tsv",
        "run_dir": "1",
    },
    "reviewer_assignment_dataset":
    {
        "eval_set_path":     f"{_reviewer_dataset_dir}/test_set.json",
        "exclude_list_path": f"{_reviewer_dataset_dir}/exclude_list.json",
        "history_length": 50,
        "include_abstracts": False,
        "max_length": 512,
        "reviewers_path":    f"{_reviewer_dataset_dir}/reviewer_publications_with_metadata.json",
        "train_qrels_path":  f"{_reviewer_dataset_dir}/train_qrels.tsv",
        "train_set_path":    f"{_reviewer_dataset_dir}/train_set.json",
        "val_qrels_path":    f"{_reviewer_dataset_dir}/val_qrels.tsv",
        "val_set_path":      f"{_reviewer_dataset_dir}/val_set.json",
    },
    "root_dir": f"{_base_dir}/runs/indexer-assignment-dataset/reviewer-assignment",
    "train": {        
        "batch_size": 40,
        "best_model_dir": "best_model",
        "config_filename": "config.json",
        "eval_accumulation_steps": 1, 
        "eval_batch_size": 80,
        "eval_steps": -1,
        "evaluation_strategy": "epoch",
        "fp16": True,
        "gradient_accumulation_steps": 3,
        "greater_is_better": False,
        "learning_rate": 3e-6,
        "load_best_model_at_end": True,
        "logging_dir": "logs",
        "logging_steps": -1,
        "logging_strategy": "epoch",
        "lr_scheduler_type": "constant_with_warmup",
        "max_grad_norm": 1.,
        "max_steps" : -1,
        "monitor_metric": "loss",
        "num_epochs": 30,
        "run_dir": None,
        "save_steps": -1,
        "save_strategy": "epoch",
        "save_total_limit": 200,
        "seed": 42,
        "warmup_steps": 28,
        "weight_decay": 0.01,
    },
}


def get():
    process_args()
    return CONFIG


def get_output_path_lookup(cfg, run_dir=None):
    if run_dir is None:
        run_dir = os.environ[SLURM_JOB_ID_ENV_VAR] if SLURM_JOB_ID_ENV_VAR in os.environ else str(int(time.time()))

    pred_cfg = cfg["pred"]
    train_cfg = cfg["train"]
    
    best_model_dir = train_cfg["best_model_dir"]
    root_dir = cfg["root_dir"]

    run_path = os.path.join(root_dir, run_dir)

    lookup = {}
    lookup["best_model_dir"] =   os.path.join(run_path, train_cfg["best_model_dir"])
    lookup["config_path"] =      os.path.join(run_path, train_cfg["config_filename"])
    lookup["logging_dir"] =      os.path.join(run_path, train_cfg["logging_dir"])
    lookup["pred_config_path"] = os.path.join(run_path, pred_cfg["config_filename"])
    lookup["results_path"] =     os.path.join(run_path, pred_cfg["results_filename"])
    lookup["run_path"] = run_path
    return lookup


def get_training_args(train_cfg, output_path_lookup):
    training_args = TrainingArguments(
        adafactor=False,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        eval_accumulation_steps=train_cfg["eval_accumulation_steps"],
        eval_steps=train_cfg["eval_steps"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        fp16=train_cfg["fp16"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        greater_is_better=train_cfg["greater_is_better"],
        learning_rate=train_cfg["learning_rate"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        logging_dir=output_path_lookup["logging_dir"],
        logging_steps=train_cfg["logging_steps"],
        logging_strategy=train_cfg["logging_strategy"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_grad_norm=train_cfg["max_grad_norm"],
        max_steps=train_cfg["max_steps"],
        metric_for_best_model=train_cfg["monitor_metric"],
        num_train_epochs=train_cfg["num_epochs"],    
        output_dir=output_path_lookup["run_path"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        per_device_train_batch_size=train_cfg["batch_size"],
        save_steps=train_cfg["save_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        warmup_steps=train_cfg["warmup_steps"],                
        weight_decay=train_cfg["weight_decay"], 
    )
    return training_args


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--best_model_dir", type=str)
    parser.add_argument("--cv_fold", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--results_filename", type=str)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    cfg = CONFIG

    if args.best_model_dir: cfg["train"]["best_model_dir"] = args.best_model_dir
    if args.cv_fold: cfg["reviewer_assignment_dataset"]["cv_fold"] = args.cv_fold
    if args.model_name: cfg["model"]["name"] = args.model_name
    if args.results_filename: cfg["pred"]["results_filename"] = args.results_filename 
    if args.run_dir:
        cfg["pred"]["run_dir"] = args.run_dir 
        cfg["train"]["run_dir"] = args.run_dir 
    if args.seed: cfg["train"]["seed"] = args.seed

    
def save(config, path):
    json.dump(config, open(path, "wt"), ensure_ascii=False, indent=4)