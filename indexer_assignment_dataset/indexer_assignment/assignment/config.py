import datetime

_dataset_dir = "/slurm_storage/raear/input-data/indexer-assignment-dataset/indexer-assignment"
_profile_matching_dir = "/slurm_storage/raear/working_dir/indexer-assignment-dataset/indexer-assignment/profile-matching"
_run_dir = "/slurm_storage/raear/runs/indexer-assignment-dataset/indexer-assignment"
_working_dir = "/slurm_storage/raear/working_dir/indexer-assignment-dataset/indexer-assignment/assignment"


CONFIG = {
    "article_dataset_path": f"{_dataset_dir}/Contractor_Test_Set.json.gz",
    "baseline_predictions_path_template": f"{_profile_matching_dir}/baseline/test-set/Contractor_Test_Set_Baseline_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "config_path_template": f"{_working_dir}/assign_config_{{type}}_{{method}}.json",
    "eval_set_end_date": datetime.date(2020,1,1).isoformat(),
    "eval_set_start_date": datetime.date(2018,5,1).isoformat(),  
    "ids_path_template": f"{_working_dir}/ids_{{type}}_{{method}}.npy",
    "indexer_profiles_1_predictions_path_template": f"{_run_dir}/29583476/results-127345/Contractor_Test_Set_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "indexer_profiles_2_predictions_path_template": f"{_run_dir}/29583479/results-127345/Contractor_Test_Set_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "indexer_profiles_3_predictions_path_template": f"{_run_dir}/29583480/results-127345/Contractor_Test_Set_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "issue_id_lookup_path": f"{_dataset_dir}/Contractor_Test_Set_Issue_Id_Lookup.csv",
    "journal_nlmids_path_template": f"{_working_dir}/journal_nlmids_{{type}}_{{method}}.npy",
    "metrics_path_template": f"{_working_dir}/assign_metrics_{{type}}_{{method}}.txt",
    "num_indexers": 102,
    "num_pools": 8,
    "pred_assignments_path_template": f"{_working_dir}/pred_assignments_{{type}}_{{method}}.npy",
    "specter_predictions_path_template": f"{_profile_matching_dir}/specter-embeddings/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Specter_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "tfidf_predictions_path_template": f"{_profile_matching_dir}/tfidf-embeddings/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Tfidf_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "true_assignments_path_template": f"{_working_dir}/true_assignments_{{type}}_{{method}}.npy",
    "weights_path_template": f"{_working_dir}/weights_{{type}}_{{method}}.npy",
}