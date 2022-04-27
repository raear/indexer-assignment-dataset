from . import config
from . import data_helper
import json
from ...shared.helper import load_qrels, make_dir
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, Trainer


def main():
    cfg = config.get()
    model_cfg = cfg["model"]
    rds_cfg = cfg["reviewer_assignment_dataset"]
    train_cfg = cfg["train"]

    cache_dir = cfg["cache_dir"]
    seed = train_cfg["seed"]

    output_path_lookup = config.get_output_path_lookup(cfg, train_cfg["run_dir"])
    make_dir(output_path_lookup["run_path"])
    config.save(cfg, output_path_lookup["config_path"])

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"], cache_dir=cache_dir)
    training_args = config.get_training_args(train_cfg, output_path_lookup)

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_cfg["name"], num_labels=2, cache_dir=cache_dir)
        return model

    train_qrels = load_qrels(rds_cfg["train_qrels_path"])
    train_queries = json.load(open(rds_cfg["train_set_path"]))
    reviewers = json.load(open(rds_cfg["reviewers_path"]))
    exclude_list = json.load(open(rds_cfg["exclude_list_path"]))
    train_dataset = data_helper.ReviewerAssignmentDataset(rds_cfg, tokenizer, False, train_qrels, train_queries, reviewers, exclude_list)
    
    val_qrels = load_qrels(rds_cfg["val_qrels_path"])
    val_queries = json.load(open(rds_cfg["val_set_path"]))
    val_dataset = data_helper.ReviewerAssignmentDataset(rds_cfg, tokenizer, True, val_qrels, val_queries, reviewers, exclude_list)
    trainer = Trainer(model_init=model_init, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=data_helper.compute_metrics)

    trainer.train()
    trainer.save_model(output_path_lookup["best_model_dir"])


if __name__ == "__main__":
    main()