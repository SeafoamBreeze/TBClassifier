from utils.optuna_utils import get_robust_median_epoch
from utils.s3_utils import download_dataset_from_s3, download_latest_optuna_study  
import mlflow
import mlflow.pytorch
from config import DATASET_PATH, STUDY_DB, TRACKING_DB
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from src.data_pipeline.data_pipeline import DataPipeline
from src.neural_network.densenet_classifier import DenseNetClassifier

pl.seed_everything(42, workers=True)
RUN_01_TRAINING_SCRIPT = os.environ.get("RUN_01_TRAINING_SCRIPT", "false").lower() == "true"
# FIT_TEST_MODEL = os.environ.get("FIT_TEST_MODEL", "false").lower() == "true"      # S3
FIT_TEST_MODEL = os.environ.get("FIT_TEST_MODEL", "true").lower() == "true"         # Local

if __name__ == "__main__":

    mlflow.set_experiment("TBClassifier_Staging")
    # mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])            # To store in S3 
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")                     # To store locally

    with mlflow.start_run():

        # download_dataset_from_s3()
        # download_latest_optuna_study()

        study = optuna.load_study(
            study_name="TBClassifier",
            storage=f"sqlite:///{STUDY_DB}"
        )

        best_params = study.best_params
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Params: {best_params}")

        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_params({
            "learning_rate": best_params["learning_rate"],
            "dropout": best_params["dropout"],
            "weight_decay": best_params["weight_decay"],
            "batch_size": best_params["batch_size"]
        })        

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
            monitor="train_loss",
            save_top_k=1,
            mode="min",
            filename="best_model"
        )
        
        trainer = pl.Trainer(
            max_epochs=get_robust_median_epoch(study),
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            enable_progress_bar=True,   
            callbacks=[checkpoint_callback],
            deterministic=True
        )

        if (FIT_TEST_MODEL):
            print("Begin model fitting")
            trainer.fit(model, datamodule)
            best_model = DenseNetClassifier.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cpu")
        else:
            print("Skipping model fitting for demo")
            best_model = model

        print("Begin model testing")
        trainer.test(best_model, datamodule)

        print("Uploading staging model to MLFLOW_ARTIFACT_ROOT in S3 as defined in CodeBuild ")
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            registered_model_name="TBClassifier_staging"
        )

        print("End of training")
