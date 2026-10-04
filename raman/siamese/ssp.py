import lightning as L
import torch
import torch.nn as nn
import logging
import numpy as np
import sys
from sklearn.linear_model import enet_path

from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy, Precision, Recall, StatScores
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics import MetricCollection

from raman.backbones.ssp import SSPModel

logger = logging.getLogger(__name__)
 
class SSPLit(L.LightningModule):
    def __init__(
        self,
        num_classes=2,
        lr=1e-3,
        weight_decay=1e-6,
        input_channels=2,
        siamese_type='cnn',
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=512,
        dim_head=64,
        pos_embedding_type='sin_cos',
        dropout=0.1,
        emb_dropout=0.1,
        fc1_size=1024,
        pretrained_ckpt=None, 
        freeze_encoder=False,
        regression=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.regression = regression 
        
        out_dim = 1 if self.regression else num_classes

        self.model = SSPModel(
            out_dim=out_dim,
            input_channels=input_channels,
            siamese_type=siamese_type,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            pos_embedding_type=pos_embedding_type,
            dropout=dropout,
            emb_dropout=emb_dropout,
            fc1_size=fc1_size,
            pretrained_ckpt=pretrained_ckpt,
            freeze_encoder=freeze_encoder
        )
        self.lr=lr
        self.weight_decay=weight_decay

        # Setup Loss and Metrics
        if self.regression:
            self.criterion = nn.MSELoss()
            self.metrics_names = ["mse", "mae", "r2"]
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.metrics_names = ["auc", "f1", "auprc", "acc", "stat"]
        
        if self.regression:
            self.val_metrics = nn.ModuleList([MeanSquaredError(), MeanAbsoluteError(), R2Score()])
            self.test_metrics = nn.ModuleList([MeanSquaredError(), MeanAbsoluteError(), R2Score()])
        else:
            print("\n\nNUM_CLASSES=", num_classes,"\n\n")
            self.val_metrics = nn.ModuleList([
                AUROC(task="multiclass", num_classes=num_classes),
                F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                AveragePrecision(task="multiclass", num_classes=num_classes),
                Accuracy(task="multiclass", num_classes=num_classes),
                StatScores(task="multiclass", num_classes=num_classes)
            ])
            self.test_metrics = nn.ModuleList([
                AUROC(task="multiclass", num_classes=num_classes),
                F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                AveragePrecision(task="multiclass", num_classes=num_classes),
                Accuracy(task="multiclass", num_classes=num_classes),
                StatScores(task="multiclass", num_classes=num_classes)
            ])
        
        self.metrics = {
            "val": (self.val_metrics, self.metrics_names),
            "test": (self.test_metrics, self.metrics_names),
        }

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]
        
        if not self.regression:
            y = y.long()
        
        logits = self(x1, x2)
        
        if self.regression:
            logits = logits.squeeze(-1)
            
        loss = self.criterion(logits, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]
        
        if not self.regression:
            y = y.long()
        
        logits = self(x1, x2)
        
        if self.regression:
            logits = logits.squeeze(-1)
            
        loss = self.criterion(logits, y)
        
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        metrics, metrics_names = self.metrics[stage]
        for idx, metric in enumerate(metrics):
            name = metrics_names[idx]
            
            if self.regression:
                metric.update(logits, y)
                self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
            else:
                if name != "stat":
                    metric.update(logits, y)
                    self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
                else:
                    metric.update(logits, y)
                    stats_ = metric.compute().detach().cpu().numpy()
                    tp = stats_[0]
                    fp = stats_[1]
                    tn = stats_[2]
                    fn = stats_[3]
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    self.log(f"{stage}_fpr", fpr, on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"{stage}_fnr", fnr, on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"{stage}_sens", sens, on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"{stage}_spec", spec, on_step=False, on_epoch=True, prog_bar=True)
            
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
            
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
        
    def predict_step(self, batch, batch_idx):
        x1 = batch["x"]
        x2 = batch["ref"]
        
        logits = self(x1, x2)
        
        output = {
            "logits": logits
        }
        
        if self.regression:
            logits = logits.squeeze(-1)
            output["preds"] = logits
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            output["probs"] = probs
            output["preds"] = preds
        
        # Include metadata if available
        for key in ["y", "component_name", "mixture_index"]:
            if key in batch:
                output[key] = batch[key]
                
        return output
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }