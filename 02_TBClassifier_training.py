from data_pipeline.data_pipeline import DataPipeline
from neural_network.densenet_classifier import DenseNetClassifier
from utils.optuna_utils import get_robust_median_epoch
from utils.s3_utils import download_latest_optuna_study, upload_file_to_s3  
import mlflow
from config import DATASET_PATH, STUDY_DB, TRACKING_TRAINING_DB, S3_BUCKET
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
import os

FIT_TEST_MODEL = os.environ.get("FIT_TEST_MODEL", "false").lower() == "true"

if __name__ == "__main__":

    mlflow_logger = MLFlowLogger(
        experiment_name="TBClassifier_PROD",
        tracking_uri=f"sqlite:///{TRACKING_TRAINING_DB}"
    )

    download_latest_optuna_study()

    study = optuna.load_study(
        study_name="TBClassifier",
        storage=f"sqlite:///{STUDY_DB}"
    )

    best_params = study.best_params
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Params: {best_params}")

    mlflow_logger.log_hyperparams(best_params)

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
        enable_progress_bar=True,   
        logger=mlflow_logger,
        deterministic=True
    )

    if (FIT_TEST_MODEL):
        print("Begin model fitting and testing")
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
    else:
        print("Skipping model fitting and testing")

    mlflow.pytorch.log_model(
        run_id=mlflow_logger.run_id,
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="tbClassifier_PROD"
    )

    upload_file_to_s3(str(TRACKING_TRAINING_DB), S3_BUCKET, "training-artifact/mlflow.db")