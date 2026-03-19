from utils.optuna_utils import run_optimization
import boto3
import mlflow
from config import TRACKING_DB

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    study = run_optimization(
        n_trials=10,
        timeout=3600*10,
        n_splits=5,
        max_epochs_per_fold=30
    )

    # TODO: Download dataset link?
    # TODO: Set up api to view optuna tuning graphs by importing the study
    # TODO: Plot the relevant graphs after training the model 

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    BUCKET = "tb-classifier-artifacts-506261418229-ap-southeast-2-an"
    s3.upload_file("optuna_studies/optuna_study.db", BUCKET, "tuning/optuna_study.db")
    s3.upload_file("mlflow.db", BUCKET, "tuning/mlflow.db")
    s3.upload_file("best_params.json", BUCKET, "tuning/best_params.json")

    # # Optional: download S3 artifacts here
    # import boto3
    # s3 = boto3.client("s3")
    # S3_BUCKET = os.environ.get("S3_BUCKET")
    # OPTUNA_DB_S3_KEY = os.environ.get("OPTUNA_DB_S3_KEY")
    # MLFLOW_DB_S3_KEY = os.environ.get("MLFLOW_DB_S3_KEY")

    # if RUN_TUNING:
    #     try:
    #         s3.download_file(S3_BUCKET, OPTUNA_DB_S3_KEY, "tuning/optuna_study.db")
    #         print("Downloaded Optuna DB from S3")
    #     except s3.exceptions.NoSuchKey:
    #         print("No Optuna DB found in S3, starting fresh")

    # if RUN_TRAINING:
    #     try:
    #         s3.download_file(S3_BUCKET, MLFLOW_DB_S3_KEY, "mlflow/tracking.db")
    #         print("Downloaded MLflow DB from S3")
    #     except s3.exceptions.NoSuchKey:
    #         print("No MLflow DB found in S3, starting fresh")