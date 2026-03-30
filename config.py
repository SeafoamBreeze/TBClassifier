from pathlib import Path

DATASET_PATH = Path("src/dataset/images")
MLFLOW_TRACKING_DIR = Path("mlflow")
OPTUNA_DIR = Path("optuna_studies")

DATASET_PATH.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_DB = MLFLOW_TRACKING_DIR/"tracking.db"
STUDY_DB = OPTUNA_DIR/"optuna_study.db"
S3_BUCKET = "tbclassifier"
S3_DATASET_PATH = Path("dataset/images")
S3_PREFIX_OPTUNA_STUDIES = "optuna_studies"
S3_PREFIX_PRODUCTION_MODEL = "production/artifacts/model"
S3_PREFIX_BUILD_ARTIFACTS = "build-artifacts"