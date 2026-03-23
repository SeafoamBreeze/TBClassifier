from utils.optuna_utils import run_optimization
from utils.s3_utils import download_model_hyperparameters
import boto3
import mlflow
from config import STUDY_DB, TRACKING_TUNING_DB, S3_BUCKET
from datetime import datetime

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_TUNING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    download_model_hyperparameters()

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3.upload_file(str(STUDY_DB), S3_BUCKET, f"tuning/{timestamp}/optuna_study.db")
        s3.upload_file(str(TRACKING_TUNING_DB), S3_BUCKET, f"tuning/{timestamp}/mlflow.db")
        s3.upload_file("best_params.json", S3_BUCKET, f"tuning/{timestamp}/best_params.json")
        s3.upload_file("best_params.json", S3_BUCKET, "training/best_params.json")
    except Exception as e:
        print(f"Upload failed: {e}")
