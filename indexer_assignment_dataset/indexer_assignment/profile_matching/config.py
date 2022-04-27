import datetime

_base_dir = "/slurm_storage/raear"
_dataset_dir = f"{_base_dir}/input-data/indexer-assignment-dataset/indexer-assignment"
_working_dir = f"{_base_dir}/working_dir/indexer-assignment-dataset/indexer-assignment/profile-matching"
_specter_dir = f"{_working_dir}/specter-embeddings"
_tfidf_dir = f"{_working_dir}/tfidf-embeddings"


CONFIG = {
    "baseline_predictions_path_template": f"{_working_dir}/baseline/test-set/Contractor_Test_Set_Baseline_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "cache_dir": "/slurm_storage/raear/cache/huggingface/transformers",
    "eval_dataset_path": f"{_dataset_dir}/Contractor_Test_Set.json.gz",
    "eval_journal_indexer_lookup_path": f"{_dataset_dir}/Contractor_Eval_Journal_Indexer_Lookup.pkl",
    "eval_set_end_date": datetime.date(2020,1,1),
    "eval_set_start_date": datetime.date(2018,5,1),  
    "num_indexers": 102,
    "pred_top_n": 50,
    "pred_use_weights": False,
    "train_set_path": f"{_dataset_dir}/Contractor_Train_Set.json.gz",
    "search_top_n": 50,
    "specter_batch_size": 4,
    "specter_document_embeddings_path_template": f"{_specter_dir}/document-embeddings/Contractor_Train_Set_Specter_Document_Embeddings_{{}}.npy",
    "specter_document_ids_path_template": f"{_specter_dir}/document-embeddings/Contractor_Train_Set_Specter_Document_Ids_{{}}.npy",
    "specter_predictions_path_template": f"{_specter_dir}/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Specter_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "specter_query_embeddings_path_template": f"{_specter_dir}/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Specter_Query_Embeddings_Year_{{year}}_Week_{{week_number}}.npy",
    "specter_query_ids_path_template": f"{_specter_dir}/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Specter_Query_Ids_Year_{{year}}_Week_{{week_number}}.npy",
    "specter_search_results_path_template": f"{_specter_dir}/test-set/year-{{year}}/week-{{week_number}}/search-results/Contractor_Test_Set_Specter_Search_Results_Year_{{year}}_Week_{{week_number}}_Indexer_Num_{{indexer_num}}.tsv",
    "tfidf_document_embeddings_path_template": f"{_tfidf_dir}/document-embeddings/Contractor_Train_Set_Tfidf_Document_Embeddings_{{}}.npz",
    "tfidf_document_ids_path_template": f"{_tfidf_dir}/document-embeddings/Contractor_Train_Set_Tfidf_Document_Ids_{{}}.npy",
    "tfidf_predictions_path_template": f"{_tfidf_dir}/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Tfidf_Predictions_Year_{{year}}_Week_{{week_number}}.tsv",
    "tfidf_query_embeddings_path_template": f"{_tfidf_dir}/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Tfidf_Query_Embeddings_Year_{{year}}_Week_{{week_number}}.npz",
    "tfidf_query_ids_path_template": f"{_tfidf_dir}/test-set/year-{{year}}/week-{{week_number}}/Contractor_Test_Set_Tfidf_Query_Ids_Year_{{year}}_Week_{{week_number}}.npy",
    "tfidf_search_results_path_template": f"{_tfidf_dir}/test-set/year-{{year}}/week-{{week_number}}/search-results/Contractor_Test_Set_Tfidf_Search_Results_Year_{{year}}_Week_{{week_number}}_Indexer_Num_{{indexer_num}}.tsv",
    "tfidf_vectorizer_path": f"{_tfidf_dir}/Contractor_Train_Set_Tfidf_Vectorizer.pkl",
}