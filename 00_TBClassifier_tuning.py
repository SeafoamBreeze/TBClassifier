from utils.optuna_utils import run_optimization
from utils.s3_utils import upload_file_to_s3
import boto3
import mlflow
from config import STUDY_DB, TRACKING_DB, S3_BUCKET
from datetime import datetime

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    study = run_optimization(
        n_trials=10,
        timeout=3600*10,
        max_epochs_per_fold=30
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_file_to_s3(str(STUDY_DB), S3_BUCKET, f"tuning/{timestamp}/optuna_study.db")
    upload_file_to_s3(str(TRACKING_DB), S3_BUCKET, f"tuning/{timestamp}/mlflow.db")
    upload_file_to_s3(str(STUDY_DB), S3_BUCKET, f"tuning/latest/optuna_studies/optuna_study.db")
    upload_file_to_s3(str(TRACKING_DB), S3_BUCKET, f"tuning/latest/mlflow/mlflow.db")
