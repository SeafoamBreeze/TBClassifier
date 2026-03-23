from utils.optuna_utils import run_optimization
from utils.s3_utils import download_dataset_from_s3, upload_file_to_s3
import boto3
import mlflow
from config import STUDY_DB, TRACKING_TUNING_DB, S3_BUCKET
from datetime import datetime

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_TUNING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    download_dataset_from_s3()

    study = run_optimization(
        n_trials=10,
        timeout=3600*10,
        n_splits=5,
        max_epochs_per_fold=30
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_file_to_s3(str(STUDY_DB), S3_BUCKET, f"tuning/{timestamp}/optuna_study.db")
    upload_file_to_s3(str(TRACKING_TUNING_DB), S3_BUCKET, f"tuning/{timestamp}/mlflow.db")
    upload_file_to_s3("best_params.json", S3_BUCKET, f"tuning/{timestamp}/best_params.json")
    upload_file_to_s3("best_params.json", S3_BUCKET, "training/best_params.json")