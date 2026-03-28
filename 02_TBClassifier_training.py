from data_pipeline.data_pipeline import DataPipeline
from neural_network.densenet_classifier import DenseNetClassifier
from utils.optuna_utils import get_robust_median_epoch
from utils.s3_utils import download_latest_optuna_study  
import mlflow
import mlflow.pytorch
from config import DATASET_PATH, STUDY_DB
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

FIT_TEST_MODEL = os.environ.get("FIT_TEST_MODEL", "false").lower() == "true"

if __name__ == "__main__":

    mlflow_logger = MLFlowLogger(
        experiment_name="TBClassifier_Staging",
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
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
        weight_decay=best_params['weight_decay'],
        tuning = False
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mcc",
        save_top_k=1,
        mode="max",
        filename="best_model"
    )
    
    trainer = pl.Trainer(
        max_epochs=get_robust_median_epoch(study),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,   
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )

    if (FIT_TEST_MODEL):
        print("Begin model fitting and testing")
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        best_model = DenseNetClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    else:
        print("Skipping model fitting and testing")
        best_model = model

    print("Uploading staging model to MLFLOW_ARTIFACT_ROOT in S3 as defined in CodeBuild ")

    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="model",
        registered_model_name="TBClassifier_staging"
    )

    print("End of training")
