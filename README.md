# The NLM Indexer Assignment Dataset

This is the GitHub repository for "The NLM indexer assignment dataset" by Alastair R. Rae, James G. Mork, and Dina Demner-Fushman.

## Download
The following downloads are provided as GitHub release assets.

### Indexer Assignment Dataset Files
Note: this GitHub repository releases indexer assignment datasets for in-house and contractor indexers. However, our paper focuses on the larger contractor indexer dataset. 

| File | Size | Format | Description
| --- | --- | --- | --- |
| Contractor_Indexer_Assignments.txt | 50Mb | PSV (pmid&#124;indexer_num) | Contractor indexer article assignments.
| In_House_Indexer_Assignments.txt | 4Mb | PSV (pmid&#124;indexer_num) | In house indexer article assignments (not discussed in paper).
| Contractor_Indexer_Dataset.json.xz | 1.4Gb | JSON (XZ archive) | Contractor indexer assignment dataset (with article metadata).
| Contractor_Indexer_Train_Set.json.gz | 1.6Gb | JSON (Gzip archive) | Contractor indexer train set.
| Contractor_Indexer_Val_Set.json.gz | 131Mb | JSON (Gzip archive) | Contractor indexer validation set.
| Contractor_Indexer_Test_Set.json.gz | 366Mb | JSON (Gzip archive) | Contractor indexer test set.
| Contractor_Indexer_Val_Set_Issue_Id_Lookup.csv | 13Mb | CSV with headers | Contractor indexer validation set article-issue id mapping file.
| Contractor_Indexer_Test_Set_Issue_Id_Lookup.csv | 35Mb | CSV with headers | Contractor indexer test set article-issue id mapping file.

### F1000research Journal Reviewer Assignment Dataset Files

| File | Size | Format | Description
| --- | --- | --- | --- |
| F1000res_assignments.json | 2.5Mb | JSON | Reviewer assignments for the F1000research journal (article metadata and reviewer ids).
| F1000res_reviewers.json | 145Mb | JSON | F1000research reviewer information (including publication list).
| F1000res_train_set.json | 1.9Mb | JSON | F1000research train set.
| F1000res_val_set.json | 253Kb | JSON | F1000research validation set.
| F1000res_test_set.json | 404Kb | JSON | F1000research test set.
| F1000res_exclude_list.json | 28Kb | JSON | For each article (identified by query_id) a list of reviewer to exclude (since they are article authors).

### Trained Models

| File | Size | Format | Description
| --- | --- | --- | --- |
| indexer_profiles_method_1.tar.gz | 1.1Gb | tar.gz | Indexer profiles model artifacts for model trained using method 1.
| indexer_profiles_method_2.tar.gz | 1.1Gb | tar.gz | Indexer profiles model artifacts for model trained using method 2.
| indexer_profiles_method_3.tar.gz | 1.1Gb | tar.gz | Indexer profiles model artifacts for model trained using method 3.
| pretrained_text_matching_model.tar.gz| 1.1Gb | tar.gz | Artifacts for reviewer expertise matching model pretrained on the indexer assignment dataset.
| fine_tuned_text_matching_model.tar.gz| 1.1Gb | tar.gz | Artifacts for reviewer expertise matching model fine-tuned on the journal reviewer assignment dataset.
| fine_tuned_pretrained_text_matching_model.tar.gz| 1.1Gb | tar.gz | Artifacts for reviewer expertise matching model pretrained on the indexer assignment dataset and then fine-tuned on the journal reviewer assignment dataset.

## Setup

### Create the Anaconda Virtual Environment:

```
conda create -n indexer-assignment
conda activate indexer-assignment
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge transformers=4.10.2
conda install -c conda-forge scikit-learn=0.24.2
conda install -c conda-forge tensorboard=2.6.0
conda install pandas=1.3.2
conda install -c conda-forge cvxpy=1.1.15
pip install pytrec_eval==0.5
```

## Usage

### Create the Indexer Assignment Dataset

1. Create a working directory.
2. Copy the lsi2022.xml and Contractor_Indexer_Assignments.txt into the working directory.
3. Set _working_dir in indexer_assignment_dataset.indexer_assignment.dataset.config.py
4. To create the dataset and associated files, run:
```
    python -m indexer_assignment_dataset.indexer_assignment.dataset.run_all
```

### Indexer Profile Matching

1. Create a working directory.
2. Set the working directory (_working_dir) and the Indexer Assignment Dataset directory (_dataset_dir) in indexer_assignment_dataset.indexer_assignment.profile_matching.config.py
4. To create test set expertise predictions for the baseline, TF-IDF, and SPECTER methods, run:
```
    python -m indexer_assignment_dataset.indexer_assignment.profile_matching.run_all
```

### Train Indexer Profiles Models

1. Set the dataset directory (_dataset_dir), the output directory (root_dir), the Hugging Face cache directory (cache_dir), and the directory for the best model weights (best_model_dir="best_model") in indexer_assignment_dataset.indexer_assignment.indexer_profiles.config.py
2. The configuration file is setup for training method #1 by default. For training method #2 set equal_expertise = True, and for training method #3 set loss_masking=True.
3. To train the model run:
```
python -m indexer_assignment_dataset.indexer_assignment.indexer_profiles.train
```

### Indexer Profiles Model Expertise Predictions

1. Set the dataset directory (_dataset_dir), the output directory (root_dir), and the Hugging Face cache directory (cache_dir) in indexer_assignment_dataset.indexer_assignment.indexer_profiles.config.py
2. Additionally, set the folder containing the model training artefacts (run_dir), and the model checkpoint to load (best_model_dir) to the model checkpoint for epoch 5 (e.g., "checkpoint-127345").
3. To make expertise predictions run:
```
python -m indexer_assignment_dataset.indexer_assignment.indexer_profiles.pred
```

### Run Assignment Algorithm

1. Set the dataset directory (_dataset_dir), the profile matching working directory (_profile_matching_dir), the indexer profiles models output directory (_run_dir), and the assignment working directory (_working_dir) in indexer_assignment_dataset.indexer_assignment.assignment.config.py
2. Additionally, configure the correct paths for the indexer profiles model predictions: indexer_profiles_1_predictions_path_template, indexer_profiles_2_predictions_path_template, indexer_profiles_3_predictions_path_template.
3. Run the assignment algorithm for all expert matching methods as follows:
```
python -m indexer_assignment_dataset.indexer_assignment.assignment.assign
```

### Run Assignment Error Analysis
1. Set the dataset directory (_dataset_dir), the assignment results directory (_assign_results_dir), and the error analysis working directory (_working_dir) in indexer_assignment_dataset.indexer_assignment.error_analysis.config.py
2. Run the journal level error analysis for all assignment results as follows:
```
python -m indexer_assignment_dataset.indexer_assignment.error_analysis.journal_level_error_analysis
```

### Create the Journal Reviewer Assignment Dataset
1. Set the working directory (working_dir), and enter your ORCID API token (orcid_api_token) in indexer_assignment_dataset.reviewer_assignment.dataset.config.py
2. To create the dataset and associated files run:
```
python -m indexer_assignment_dataset.reviewer_assignment.dataset.run_all
```

### Reviewer Profile Matching
1. Set the working directory (_working_dir) and the Journal Reviewer Assignment Dataset directory (_dataset_dir) in indexer_assignment_dataset.reviewer_assignment.profile_matching.config.py
2. Additionally, set the Hugging Face cache direcotry (cache_dir), and specify the location of the TF-IDF model (tfidf_model_path) that was created for indexer profile matching.
3. Create reviewer recommendations for the test set by running the following:
```
   python -m indexer_assignment_dataset.reviewer_assignment.profile_matching.run_all
```

### Pretrain Text Matching Model on the Indexer Assignment Dataset
1. Set the indexer assignment dataset directory (_indexer_dataset_dir), the Hugging Face cache directory (cache_dir), and the output directory (root_dir) in indexer_assignment_dataset.reviewer_assignment.transfer_learning.config.py
2. Make the following additional modifications to the configuration: batch_size=32, eval_batch_size=64, eval_steps=1000, evaluation_strategy="steps", gradient_accumulation_steps=4, logging_steps=100, logging_strategy="steps", lr_scheduler_type="linear", num_epochs=1, save_steps=1000, save_strategy= "steps", warmup_steps=1000.
3. Run model pretraining as follows:
```
python -m indexer_assignment_dataset.reviewer_assignment.transfer_learning.indexer_assignment_train
```

### Fine-Tune Text Matching Model on the Journal Reviewer Assignment Dataset
1. Set the reviewer assignment dataset directory (_reviewer_dataset_dir), the Hugging Face cache directory (cache_dir), and the output directory (root_dir) in indexer_assignment_dataset.reviewer_assignment.transfer_learning.config.py
2. To start fine-tuning from indexer assignment dataset pretrained weights set cfg["model"]["name"] to the path of the pretrained checkpoint (e.g., "/slurm_storage/raear/runs/indexer-assignment-dataset/indexer-assignment/29652016/checkpoint-20000") otherwise use "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" for PubMedBERT weights.
3. Run model fine-tuning as follows:
```
python -m indexer_assignment_dataset.reviewer_assignment.transfer_learning.reviewer_assignment_train
```

### Text Matching Model Reviewer Expertise Predictions
1. Set the reviewer assignment dataset directory (_reviewer_dataset_dir), the Hugging Face cache directory (cache_dir), and the output directory (root_dir) in indexer_assignment_dataset.reviewer_assignment.transfer_learning.config.py
2. Additionally, set run_dir to the folder containing the fine-tuned model artefacts, and set best_model_dir to directory of the model checkpoint with the highest validation set F1 score (e.g. checkpoint-84).
3. Make reviewer expertise predictions as follows:
```
python -m indexer_assignment_dataset.reviewer_assignment.transfer_learning.reviewer_assignment_pred
```