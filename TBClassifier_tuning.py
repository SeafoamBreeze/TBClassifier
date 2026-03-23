from utils.optuna_utils import run_optimization
from utils.s3_utils import download_dataset_from_s3, upload_file_to_s3
import boto3
import mlflow
from config import STUDY_DB, TRACKING_DB, BUCKET, DATASET_PATH, S3_DATASET_PATH

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    download_dataset_from_s3()

    study = run_optimization(
        n_trials=10,
        timeout=3600*10,
        n_splits=5,
        max_epochs_per_fold=30
    )

    # TODO: Set up api to view optuna tuning graphs by importing the study
    # TODO: Plot the relevant graphs after training the model 

    upload_file_to_s3(STUDY_DB, BUCKET, "tuning/optuna_study.db")
    upload_file_to_s3(TRACKING_DB, BUCKET, "tuning/mlflow.db")
    upload_file_to_s3("best_params.json", BUCKET, "tuning/best_params.json")
    upload_file_to_s3("best_params.json", BUCKET, "training/best_params.json")

