from pathlib import Path

DATASET_PATH = Path("dataset/images")
MLFLOW_TRACKING_DIR = Path("mlflow")
OPTUNA_DIR = Path("optuna_studies")

DATASET_PATH.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_TUNING_DB = MLFLOW_TRACKING_DIR/"tracking_tuning.db"
TRACKING_TRAINING_DB = MLFLOW_TRACKING_DIR/"tracking_training.db"

STUDY_DB = OPTUNA_DIR/"optuna_study.db"
S3_BUCKET = "tbclassifier-build-artifacts"
S3_DATASET_PATH = Path("dataset/images")
