from pathlib import Path

TB_CLASSIFIER_MAIN_DIR = Path("C:/Users/zheng/Desktop/TBClassifier")
DATASET_PATH = TB_CLASSIFIER_MAIN_DIR/"dataset/images"

MLFLOW_TRACKING_DIR = TB_CLASSIFIER_MAIN_DIR/"mlflow"
TRACKING_DB = MLFLOW_TRACKING_DIR/"tracking.db"

OPTUNA_DIR = TB_CLASSIFIER_MAIN_DIR/"optuna_studies"
STUDY_DB = OPTUNA_DIR/"optuna_study.db"

DATASET_PATH.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

BUCKET = "tb-classifier-artifacts-506261418229-ap-southeast-2-an"

S3_DATASET_PATH = Path("dataset/images")