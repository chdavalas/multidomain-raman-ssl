import lightning as L
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy, StatScores
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

logger = logging.getLogger(__name__)

class BaseLit(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.regression = kwargs.get("regression", False)
        self.num_classes = kwargs.get("num_classes")
        self.lr = kwargs.get("lr")
        self.weight_decay = kwargs.get("weight_decay")
        self.input_channels = kwargs.get("input_channels")

        kwargs["output_dim"] = 1 if self.regression else self.num_classes
        self.net = self._build_model(**kwargs)

        # Conditionally setup Loss and Metrics
        if self.regression:
            self.criterion = nn.MSELoss()
            self.metrics_names = ["mse", "mae", "r2"]
            
            self.val_metrics = nn.ModuleList([
                MeanSquaredError(), MeanAbsoluteError(), R2Score()
            ])
            self.test_metrics = nn.ModuleList([
                MeanSquaredError(), MeanAbsoluteError(), R2Score()
            ])
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.metrics_names = ["auc", "f1", "auprc", "acc", "stat"]
            
            def make_cls_metrics():
                return nn.ModuleList([
                    AUROC(task="multiclass", num_classes=self.num_classes),
                    F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
                    AveragePrecision(task="multiclass", num_classes=self.num_classes),
                    Accuracy(task="multiclass", num_classes=self.num_classes),
                    StatScores(task="multiclass", num_classes=self.num_classes)
                ])
                
            self.val_metrics = make_cls_metrics()
            self.test_metrics = make_cls_metrics()

        self.metrics = {
            "val": (self.val_metrics, self.metrics_names),
            "test": (self.test_metrics, self.metrics_names),
        }

    def _build_model(self, **kwargs) -> nn.Module: 
        raise ValueError("You must implement the _build_model method in your subclass")

    def _prepare_data(self, x, y):
        if self.input_channels == 1:
            x = x[:, -1, :].unsqueeze(1) 
        return x, y

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch.get("x")
        y = batch.get("y")
        x, y = self._prepare_data(x, y)

        logits = self(x)
        
        # Format the logits shape for regression
        if self.regression:
            logits = logits.squeeze(-1)

        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        x = batch.get("x")
        y = batch.get("y")
        x, y = self._prepare_data(x, y)

        logits = self(x)
        
        if self.regression:
            logits = logits.squeeze(-1)
            
        loss = self.criterion(logits, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics, metrics_names = self.metrics[stage]

        for idx, metric in enumerate(metrics):
            name = metrics_names[idx]
            
            # Regression metrics are straightforward updates
            if self.regression:
                metric.update(logits, y)
                self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
                
            # Classification requires special handling for 'stat'
            else:
                if name != "stat":
                    metric.update(logits, y)
                    self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
                else:
                    metric.update(logits, y)
                    stats_ = metric.compute().detach().cpu().numpy()
                    tp, fp, tn, fn = stats_[0], stats_[1], stats_[2], stats_[3]
                    
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
        x = batch.get("x")
        y = batch.get("y")
        x, y = self._prepare_data(x, y)

        logits = self(x)

        if self.regression:
            # Return continuous physical values
            preds = logits.squeeze(-1)
        else:
            # Return predicted class indices
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss" if self.regression else "val_loss", 
        }