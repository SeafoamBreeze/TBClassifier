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

        self.train_accs = []
        self.train_losses = []
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

        self.train_accs.append(acc.detach())
        self.train_losses.append(loss.detach())

        return loss

    def on_train_epoch_end(self):

        avg_acc = torch.stack(self.train_accs).mean()
        avg_loss = torch.stack(self.train_losses).mean()
        
        self.log('train_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True,prog_bar=True)

        self.train_accs.clear()
        self.train_losses.clear()

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

    def on_validation_epoch_end(self):

        all_preds  = torch.cat(self.val_preds)
        all_targets  = torch.cat(self.val_targets)

        try:
            mcc = matthews_corrcoef(all_targets.numpy(), all_preds.numpy())
        except ValueError:
            mcc = 0.0

        self.log("val_mcc", mcc, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        loss = F.cross_entropy(logits, y, weight=self.class_weights)

        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):

        all_preds = torch.cat(self.test_preds)
        all_targets = torch.cat(self.test_targets)
        y_pred = all_preds.numpy()
        y_true = all_targets.numpy()
        tb_class = 2
        
        metrics = {
            'test_TB_precision': precision_score(y_true, y_pred, labels=[tb_class], average=None)[0],
            'test_TB_recall': recall_score(y_true, y_pred, labels=[tb_class], average=None)[0],
            'test_TB_f2': fbeta_score(y_true, y_pred, beta=2, labels=[tb_class], average=None)[0],
            'test_mcc': matthews_corrcoef(y_true, y_pred),
        }
        
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        self.test_preds.clear()
        self.test_targets.clear()

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
