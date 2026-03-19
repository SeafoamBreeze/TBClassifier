import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import precision_score, recall_score, fbeta_score, matthews_corrcoef


class DenseNetClassifier(pl.LightningModule):
    def __init__(self, learning_rate, dropout, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = models.densenet121(weights="IMAGENET1K_V1")

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
        self.register_buffer('class_weights', class_weights)

        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True,prog_bar=True)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):

        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        mcc = matthews_corrcoef(targets.numpy(), preds.numpy())
        self.log("val_mcc", mcc, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        tb_class = 2
        tb_precision = precision_score(y.cpu().numpy(), preds.cpu().numpy(), labels=[tb_class], average="macro", zero_division=0)
        tb_recall = recall_score(y.cpu().numpy(), preds.cpu().numpy(), labels=[tb_class], average="macro", zero_division=0)
        tb_f2 = fbeta_score(y.cpu().numpy(), preds.cpu().numpy(), beta=2, labels=[tb_class], average="macro", zero_division=0)
        try:
            mcc = matthews_corrcoef(y.cpu().numpy(), preds.cpu().numpy())
        except Exception:
            mcc = 0.0

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_TB_precision", tb_precision)
        self.log("test_TB_recall", tb_recall)
        self.log("test_TB_f2", tb_f2)
        self.log("test_mcc", mcc)

        return {
            "test_loss": loss.detach(),
            "test_acc": acc.detach(),
            "test_TB_precision": torch.tensor(tb_precision, device=y.device),
            "test_TB_recall": torch.tensor(tb_recall, device=y.device),
            "test_TB_f2": torch.tensor(tb_f2, device=y.device),
            "test_mcc": torch.tensor(mcc, device=y.device)
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }
