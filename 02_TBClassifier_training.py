from data_pipeline.data_pipeline import DataPipeline
from neural_network.densenet_classifier import DenseNetClassifier
from utils.optuna_utils import get_robust_median_epoch
from utils.s3_utils import download_latest_optuna_study
import boto3
import mlflow
from config import DATASET_PATH, STUDY_DB, TRACKING_TRAINING_DB, S3_BUCKET
from datetime import datetime
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

if __name__ == "__main__":

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_TRAINING_DB}")
    mlflow.set_experiment("TBClassifier_tuning")

    # TODO: Model Registry
    # TODO: Generate sklearn metric without training?
    mlflow.pytorch.autolog(registered_model_name="TBClassifier_Best_Model")

    download_latest_optuna_study()

    study = optuna.load_study(
        study_name="TBClassifier",
        storage=f"sqlite:///{STUDY_DB}"
    )

    best_params = study.best_params
    print(f"Best Params: {best_params}")

    datamodule = DataPipeline(
        dataset_dir=DATASET_PATH,
        batch_size=best_params['batch_size'],
        tuning = False
    )

    model = DenseNetClassifier(
        learning_rate=best_params['learning_rate'],
        dropout=best_params['dropout'],
        weight_decay=best_params['weight_decay']
    )
    
    trainer = pl.Trainer(
        max_epochs=get_robust_median_epoch(study),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[],
        enable_progress_bar=True,
        logger=MLFlowLogger(
            experiment_name="TBClassifier_best_model",
            tracking_uri=f"sqlite:///{TRACKING_TRAINING_DB}"
        ),
        deterministic=True
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

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
