import argparse
import datetime
import json
import os.path
from transformers import TrainingArguments


_base_dir = "/slurm_storage/raear"
_dataset_dir = f"{_base_dir}/input-data/indexer-assignment-dataset/indexer-assignment"


CONFIG = {
    "cache_dir": f"{_base_dir}/cache/huggingface/transformers",
    "dataset": {
        "equal_expertise": False,
        "eval_dataset_path": f"{_dataset_dir}/Contractor_Test_Set.json.gz",
        "journal_indexer_lookup_path": f"{_dataset_dir}/Contractor_Journal_Indexer_Lookup.pkl",
        "loss_masking": False,
        "max_length": 512,
        "num_indexers": 102,
        "train_set_path": f"{_dataset_dir}/Contractor_Train_Set.json.gz",
        "val_limit": 40000,
        "val_set_path": f"{_dataset_dir}/Contractor_Val_Set.json.gz",
    },
    "model": {
        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "tokenizer_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    },
    "pred": {
        "eval_set_end_date": datetime.date(2020,1,1),
        "eval_set_start_date": datetime.date(2018,5,1),  
        "run_dir": "29583480",
        "results_filename": "results-127345/Contractor_Test_Set_Predictions_Year_{year}_Week_{week_number}.tsv",
    },
    "root_dir": f"{_base_dir}/runs/indexer-assignment-dataset/indexer-assignment",
    "train": {        
        "batch_size": 40, 
        "best_model_dir": "checkpoint-127345",
        "config_filename": "config.json",
        "eval_accumulation_steps": 1, 
        "eval_batch_size": 80,
        "eval_steps": -1,
        "evaluation_strategy": "epoch",
        "fp16": True,
        "gradient_accumulation_steps": 3,
        "greater_is_better": False,
        "learning_rate": 3e-5,
        "logging_dir": "logs",
        "logging_steps": -1,
        "logging_strategy": "epoch",
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.,
        "max_steps" : -1,
        "monitor_metric": "loss",
        "num_epochs": 5,
        "run_dir": None,
        "save_steps": -1,
        "save_strategy": "epoch",
        "save_total_limit": 50,
        "warmup_steps": 10000,
        "weight_decay": 0.01,
    },
}


def get():
    process_args()
    return CONFIG


def get_output_path_lookup(cfg, run_dir):
    pred_cfg = cfg["pred"]
    train_cfg = cfg["train"]
    
    run_path = os.path.join(cfg["root_dir"], run_dir)

    lookup = {}
    lookup["best_model_dir"] = os.path.join(run_path, train_cfg["best_model_dir"])
    lookup["config_path"] =    os.path.join(run_path, train_cfg["config_filename"])
    lookup["logging_dir"] =    os.path.join(run_path, train_cfg["logging_dir"])
    lookup["results_path"] =   os.path.join(run_path, pred_cfg["results_filename"])
    lookup["run_path"] = run_path
    return lookup


def get_training_args(train_cfg, output_path_lookup):
    training_args = TrainingArguments(
        adafactor=False,
        dataloader_num_workers=1,
        eval_accumulation_steps=train_cfg["eval_accumulation_steps"],
        eval_steps=train_cfg["eval_steps"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        fp16=train_cfg["fp16"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        greater_is_better=train_cfg["greater_is_better"],
        learning_rate=train_cfg["learning_rate"],
        load_best_model_at_end=True,
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
        warmup_steps=train_cfg["warmup_steps"],                
        weight_decay=train_cfg["weight_decay"], 
    )
    return training_args


def process_args():

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--equal_expertise", type=str2bool)
    parser.add_argument("--loss_masking", type=str2bool)

    args = parser.parse_args()

    cfg = CONFIG
    ds_cfg = cfg["dataset"]

    if args.equal_expertise is not None: ds_cfg["equal_expertise"] = args.equal_expertise
    if args.loss_masking is not None: ds_cfg["loss_masking"] = args.loss_masking


def save(config, path):
    json.dump(config, open(path, "wt"), ensure_ascii=False, indent=4)