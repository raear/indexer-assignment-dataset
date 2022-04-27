from . import config
from . import data_helper
import gzip
import json
from . import modeling_bert
from scipy.special import expit
from ..shared.helper import enumerate_year_week_numbers, filter_dataset_by_weeknumber
from ...shared.helper import make_parent_dir, save_run
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

    training_args = config.get_training_args(train_cfg, output_path_lookup)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"], cache_dir=cache_dir)
    
    model = modeling_bert.BertForMultiLabelClassification.from_pretrained(output_path_lookup["best_model_dir"], cache_dir=cache_dir)

    eval_dataset = json.load(gzip.open(ds_cfg["eval_dataset_path"], mode="rt", encoding="utf8"))
   
    for year, week_number in enumerate_year_week_numbers(pred_cfg["eval_set_start_date"], pred_cfg["eval_set_end_date"]):
        
        filtered_dataset = filter_dataset_by_weeknumber(eval_dataset, year, week_number)
        if len(filtered_dataset) == 0:
            continue
        pred_dataset = data_helper.PredDataset(ds_cfg, tokenizer, filtered_dataset)

        trainer = Trainer(model=model, args=training_args)
        trainer.remove_callback(TensorBoardCallback)
        result = trainer.predict(pred_dataset)
        predictions = result.predictions
        predictions = expit(predictions)

        run = {}
        for i, example in enumerate(filtered_dataset):
            q_id = example["pmid"]
            run[q_id] = {}
            for j in range(ds_cfg["num_indexers"]):
                indexer_num = j + 1
                score = predictions[i][j]
                run[q_id][indexer_num] = score
    
        results_path = output_path_lookup["results_path"].format(year=year, week_number=week_number)
        make_parent_dir(results_path)
        save_run(results_path, run)


if __name__ == "__main__":
    main()