_assign_results_dir = "/slurm_storage/raear/working_dir/indexer-assignment-dataset/indexer-assignment/assignment"
_dataset_dir = "/slurm_storage/raear/input-data/indexer-assignment-dataset/indexer-assignment"
_working_dir = "/slurm_storage/raear/working_dir/indexer-assignment-dataset/indexer-assignment/error-analysis"


CONFIG = {
    "error_analysis_path": f"{_working_dir}/error_analysis_{{type}}_{{method}}.csv",
    "error_analysis_summary_path": f"{_working_dir}/error_analysis_summary_{{type}}_{{method}}.csv",
    "indexer_journal_descriptor_lookup_path": f"{_dataset_dir}/Contractor_Indexer_Journal_Descriptor_Lookup.pkl",
    "journal_descriptor_lookup_path": f"{_dataset_dir}/Journal_Descriptor_Lookup.pkl",
    "journal_indexer_lookup_path": f"{_dataset_dir}/Contractor_Eval_Journal_Indexer_Lookup.pkl",
    "journal_nlmids_path_template": f"{_assign_results_dir}/journal_nlmids_{{type}}_{{method}}.npy",
    "pred_assignments_path_template": f"{_assign_results_dir}/pred_assignments_{{type}}_{{method}}.npy",
    "true_assignments_path_template": f"{_assign_results_dir}/true_assignments_{{type}}_{{method}}.npy",
    "weights_path_template": f"{_assign_results_dir}/weights_{{type}}_{{method}}.npy",
}