from config import DATASET_PATH, STUDY_DB, TRACKING_DB
from src.data_pipeline.data_pipeline import DataPipeline
import mlflow
from src.neural_network.densenet_classifier import DenseNetClassifier
import optuna
import mlflow
from mlflow.tracking import MlflowClient
from utils.optuna_utils import get_robust_median_epoch
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}" )
client = MlflowClient()
experiment = client.get_experiment_by_name("TBClassifier_Staging")
print(experiment)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

# print(runs)
# print(f"Run ID: {runs[0].info.run_id}")
# print(f"  Start time: {runs[0].info.start_time}")
# print(f"  Status: {runs[0].info.status}")
# print(f"  Metrics: {runs[0].data.metrics}")
# print(f"  Params: {runs[0].data.params}")
# print(f"  Artifact URI: {runs[0].info.artifact_uri}")

study = optuna.load_study(
    study_name="TBClassifier",
    storage=f"sqlite:///{STUDY_DB}"
)

best_params = study.best_params
model = mlflow.pytorch.load_model(f"runs:/{runs[0].info.run_id}/model")
datamodule = DataPipeline(
    dataset_dir=DATASET_PATH,
    batch_size=best_params['batch_size'],
    tuning = False
)

trainer = pl.Trainer(
    max_epochs=get_robust_median_epoch(study),
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    enable_progress_bar=True, 
    deterministic=True
)
trainer.test(model, datamodule)
print("Training model loaded successfully!")    
