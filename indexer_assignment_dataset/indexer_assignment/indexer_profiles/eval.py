from . import config
from . import data_helper
from . import modeling_bert
from transformers import AutoTokenizer, Trainer
from transformers.integrations import TensorBoardCallback


def main():
    cfg = config.get()
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    pred_cfg = cfg["pred"]
    train_cfg = cfg["train"]

    cache_dir = cfg["cache_dir"]
    output_path_lookup = config.get_output_path_lookup(cfg, pred_cfg["run_dir"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"], cache_dir=cache_dir)
    training_args = config.get_training_args(train_cfg, output_path_lookup)
    
    model = modeling_bert.BertForMultiLabelClassification.from_pretrained(output_path_lookup["best_model_dir"], cache_dir=cache_dir)

    val_dataset = data_helper.create_dataset(ds_cfg, tokenizer, is_val=True, limit=pred_cfg["eval_limit"])

    trainer = Trainer(model=model, args=training_args, eval_dataset=val_dataset, compute_metrics=data_helper.compute_metrics)
    trainer.remove_callback(TensorBoardCallback)
    result = trainer.evaluate()
    print(result)


if __name__ == "__main__":
    main()