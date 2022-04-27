_base_dir = "/slurm_storage/raear/working_dir/indexer-assignment-dataset/reviewer-assignment"
_dataset_dir = f"{_base_dir}/dataset"
_working_dir = f"{_base_dir}/profile-matching"


CONFIG = {
    "cache_dir": "/slurm_storage/raear/cache/huggingface/transformers",
    "documents_path": f"{_dataset_dir}/documents.tsv",
    "exclude_list_path": f"{_dataset_dir}/exclude_list.json",
    "pred_top_n": 3,
    "queries_path": f"{_dataset_dir}/test_queries.tsv",
    "reviewers_path": f"{_dataset_dir}/reviewers.tsv",
    "search_top_n": 100000000,
    "specter_batch_size": 4,
    "specter_document_embeddings_path": f"{_working_dir}/specter_document_embeddings.npy",
    "specter_predictions_path": f"{_working_dir}/specter_test_set_predictions.tsv",
    "specter_query_embeddings_path": f"{_working_dir}/specter_test_set_query_embeddings.npy",
    "specter_search_results_path": f"{_working_dir}/specter_test_set_search_results.tsv",
    "tfidf_document_embeddings_path": f"{_working_dir}/tfidf_document_embeddings.npz",
    "tfidf_model_path": f"/slurm_storage/raear/working_dir/indexer-assignment-dataset/indexer-assignment/profile-matching/tfidf-embeddings/Contractor_Train_Set_Tfidf_Vectorizer.pkl",
    "tfidf_predictions_path": f"{_working_dir}/tfidf_test_set_predictions.tsv",
    "tfidf_query_embeddings_path": f"{_working_dir}/tfidf_test_set_query_embeddings.npz",
    "tfidf_search_results_path": f"{_working_dir}/tfidf_test_set_search_results.tsv",
}