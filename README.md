# The NLM Indexer Assignment Dataset

This is the GitHub repository for "The NLM indexer assignment dataset" by Alastair R. Rae, James G. Mork, and Dina Demner-Fushman.

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