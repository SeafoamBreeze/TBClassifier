from utils.optuna_utils import run_optimization
from utils.s3_utils import download_dataset
import boto3
import mlflow
from config import STUDY_DB, TRACKING_DB, BUCKET, DATASET_PATH, S3_DATASET_PATH

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    # TODO: Download model's hyperparameters
    # download_dataset(BUCKET, str(DATASET_PATH/"train"), str(DATASET_PATH/"train"))
    # download_dataset(BUCKET, str(DATASET_PATH/"test"), str(DATASET_PATH/"test"))

    # Final Training
    study = run_optimization(
        n_trials=10,
        timeout=3600*10,
        n_splits=5,
        max_epochs_per_fold=30
    )

    # TODO: Set up api to view optuna tuning graphs by importing the study
    # TODO: Plot the relevant graphs after training the model 

    s3 = boto3.client("s3")

    try:
        s3.upload_file(STUDY_DB, BUCKET, "tuning/optuna_study.db")
        s3.upload_file(TRACKING_DB, BUCKET, "tuning/mlflow.db")
        s3.upload_file("best_params.json", BUCKET, "tuning/best_params.json")
        s3.upload_file("best_params.json", BUCKET, "training/best_params.json")
    except Exception as e:
        print(f"Upload failed: {e}")
