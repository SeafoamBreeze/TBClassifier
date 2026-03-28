import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import precision_score, recall_score, fbeta_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pandas as pd

class DenseNetClassifier(pl.LightningModule):

    def __init__(self, learning_rate, dropout, weight_decay, tuning):

        super().__init__()
        self.save_hyperparameters()
        self.backbone = models.densenet121(weights="IMAGENET1K_V1")
        self.tuning = tuning

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(512, 3)
        )

        # Normalized Inverse Frequency for 3800 Healthy, 3800 SickNonTB, 800 TB
        class_weights = torch.tensor([1.00, 1.00, 4.75])
        self.register_buffer("class_weights", class_weights)

        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        mlflow.log_metric("train_acc", acc.item(), step=self.global_step)
        mlflow.log_metric("train_loss", loss.item(), step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        mlflow.log_metric("val_acc_step", acc.item(), step=self.global_step)
        mlflow.log_metric("val_loss_step", loss.item(), step=self.global_step)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self):

        all_preds  = torch.cat(self.val_preds)
        all_targets  = torch.cat(self.val_targets)
        y_pred = all_preds.numpy()
        y_true = all_targets.numpy()

        mcc = matthews_corrcoef(y_true, y_pred) if len(y_true) > 0 else 0.0
        mlflow.log_metric("val_mcc", mcc)

        self.val_preds = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        mlflow.log_metric("test_acc", acc.item(), step=self.global_step)
        mlflow.log_metric("test_loss", loss.item(), step=self.global_step)
        
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_start(self):
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self):

        all_preds = torch.cat(self.test_preds)
        all_targets = torch.cat(self.test_targets)
        y_pred = all_preds.numpy()
        y_true = all_targets.numpy()

        self.save_predictions(y_pred, y_true)

        tb_class = 2
        metrics = {
            "test_TB_precision": precision_score(y_true, y_pred, labels=[tb_class], average=None)[0],
            "test_TB_recall": recall_score(y_true, y_pred, labels=[tb_class], average=None)[0],
            "test_TB_f2": fbeta_score(y_true, y_pred, beta=2, labels=[tb_class], average=None)[0],
            "test_mcc": matthews_corrcoef(y_true, y_pred),
        }
        
        for name, value in metrics.items():
             mlflow.log_metric(name, value)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.create_plot(cm, normalize=False)
        self.create_plot(cm, normalize=True)

        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        if self.tuning:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        
        return optimizer

    def create_plot(self, cm, normalize):

        class_labels = ["Health", "SickNonTb", "TB"]

        if normalize:
            title = "Confusion Matrix (Normalized)"
            output_name = "confusion_matrix_normalized.png"
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]            
            fmt = ".2f"
        else:
            title = "Confusion Matrix"
            output_name = "confusion_matrix.png"
            fmt = "d"
        
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            square=True,
            linewidths=1,
            linecolor="white",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={"shrink": 0.8}
        )
        
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=15)
        ax.set_ylabel("True Label", fontsize=12, labelpad=15)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        mlflow.log_figure(fig, output_name)
        fig.savefig(output_name, dpi=300)
        plt.close(fig)

    def save_predictions(self, y_pred, y_true):
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        file_path = "predictions.csv"
        df.to_csv(file_path, index=False)
        mlflow.log_artifact(file_path, artifact_path="predictions")