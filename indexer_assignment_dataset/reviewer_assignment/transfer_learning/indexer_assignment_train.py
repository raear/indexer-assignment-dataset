from . import config
from . import data_helper
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


def main():
    cfg = config.get()
    ds_cfg = cfg["indexer_assignment_dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    cache_dir = cfg["cache_dir"]

    output_path_lookup = config.get_output_path_lookup(cfg)
    Path(output_path_lookup["run_path"]).mkdir(parents=False, exist_ok=True)
    config.save(cfg, output_path_lookup["config_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"], cache_dir=cache_dir)
    training_args = config.get_training_args(train_cfg, output_path_lookup)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["name"], num_labels=2, cache_dir=cache_dir)

    train_dataset = data_helper.create_dataset(ds_cfg, tokenizer, is_val=False)
    val_dataset   = data_helper.create_dataset(ds_cfg, tokenizer, is_val=True)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=data_helper.compute_metrics)
    trainer.train()
    trainer.save_model(output_path_lookup["best_model_dir"])


if __name__ == "__main__":
    main()