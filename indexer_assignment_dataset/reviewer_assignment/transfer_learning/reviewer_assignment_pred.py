from . import config
from . import data_helper
import json
from scipy.special import softmax
from ...shared.helper import save_run
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from transformers.integrations import TensorBoardCallback


def main():
    cfg = config.get()
    pred(cfg)


def pred(cfg):
    model_cfg = cfg["model"]
    pred_cfg = cfg["pred"]
    rds_cfg = cfg["reviewer_assignment_dataset"]
    train_cfg = cfg["train"]

    cache_dir = cfg["cache_dir"]

    output_path_lookup = config.get_output_path_lookup(cfg, pred_cfg["run_dir"])
    
    config.save(cfg, output_path_lookup["pred_config_path"])

    exclude_list = json.load(open(rds_cfg["exclude_list_path"]))

    model = AutoModelForSequenceClassification.from_pretrained(output_path_lookup["best_model_dir"], num_labels=2, cache_dir=cache_dir)
    training_args = config.get_training_args(train_cfg, output_path_lookup)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"], cache_dir=cache_dir)

    queries = json.load(open(rds_cfg["eval_set_path"]))
    queries = queries[:pred_cfg["limit"]]
    reviewers = json.load(open(rds_cfg["reviewers_path"]))
    reviewer_count = len(reviewers)
    
    dataset = data_helper.ReviewerAssignmentPredDataset(rds_cfg, tokenizer, True, queries, reviewers)

    trainer = Trainer(model=model, args=training_args)
    trainer.remove_callback(TensorBoardCallback)
    result = trainer.predict(dataset)
    predictions = result.predictions
    predictions = softmax(predictions, axis=1)

    run = {}
    for q_idx, query in enumerate(queries):
        q_id = query["query_id"]
        run[q_id] = {}
        for r_idx, reviewer in enumerate(reviewers):
            r_id = reviewer["reviewer_id"]
            if r_id in exclude_list[str(q_id)]:
                print(f"Excluded r_id, {r_id}, for q_id, {q_id}.")
                continue
            pred_idx = (q_idx * reviewer_count) + r_idx
            score = predictions[pred_idx][1]
            run[q_id][r_id] = score
 
    save_run(output_path_lookup["results_path"], run)


if __name__ == "__main__":
    main()