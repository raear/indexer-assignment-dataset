from . import config
from . import data_helper
from . import modeling_bert
import os
import time
from transformers import AutoConfig, AutoTokenizer, Trainer


SLURM_JOB_ID_ENV_VAR = "SLURM_JOB_ID"


def main():
    cfg = config.get()
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    cache_dir = cfg["cache_dir"]
    run_dir = train_cfg["run_dir"]
 
    if not run_dir:
        run_dir = os.environ[SLURM_JOB_ID_ENV_VAR] if SLURM_JOB_ID_ENV_VAR in os.environ else str(int(time.time()))

    output_path_lookup = config.get_output_path_lookup(cfg, run_dir)

    run_path = output_path_lookup["run_path"]
    if not os.path.isdir(run_path):
        os.mkdir(run_path)
    config.save(cfg, output_path_lookup["config_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"], cache_dir=cache_dir)
    training_args = config.get_training_args(train_cfg, output_path_lookup)
    
    auto_config = AutoConfig.from_pretrained(model_cfg["name"], num_labels=ds_cfg["num_indexers"], cache_dir=cache_dir)
    model = modeling_bert.BertForMultiLabelClassification.from_pretrained(model_cfg["name"], config=auto_config, cache_dir=cache_dir)

    train_dataset = data_helper.create_dataset(ds_cfg, tokenizer, is_val=False)
    val_dataset   = data_helper.create_dataset(ds_cfg, tokenizer, is_val=True, limit=ds_cfg["val_limit"])

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=data_helper.compute_metrics)
    trainer.train()
    trainer.save_model(output_path_lookup["best_model_dir"])


if __name__ == "__main__":
    main()