import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from raman.backbones.spectra_encoder import spectra_encoder
import torch.nn.functional as F
import logging
import math
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassStatScores,
)

logger = logging.getLogger(__name__)

class SpectraEncoderClassifierLit(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            lr: float = 1e-4,
            weight_decay: float = 1e-6,
            input_channels: int = 2,
            d_model: int = 64,
            encoder_depth: int = 1,
            nhead: int = 1,
            mlp_dim: int = 256,
            dim_head: int = 64,
            hidden_dim: int = 256,
            pos_embedding_type: str = 'sin_cos',
            dropout: float = 0.,
            emb_dropout: float = 0.,
            cls_token: bool = False,
            classification_head: str = 'linear',
            task: str = "multiclass",
            weighted=False
        ):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.task = task
        self.net = spectra_encoder(
                input_dim=input_channels,
                dim=d_model,
                depth=encoder_depth,
                heads=nhead,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                pos_embedding_type=pos_embedding_type,
                dropout=dropout,
                emb_dropout=emb_dropout,
                )

        if task == 'multilabel':
            self.num_labels = 4
            self.num_levels = num_classes 
            if classification_head == 'linear':
                self.classification_head = nn.Linear(
                    d_model,
                    self.num_labels * self.num_levels
                )
            elif classification_head == 'mlp':
                self.classification_head = nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.num_labels * self.num_levels)
                )
            else:
                logger.info("No valid classification head was chosen.\nValid options for classification head: linear, mlp")
                
            def make_metrics(include_stats=False, weighted=False):
                if weighted:
                    metrics_list = [
                        AUROC(task="multiclass", num_classes=self.num_levels, average="weighted"),
                        F1Score(task="multiclass", num_classes=self.num_levels, average="weighted"),
                        AveragePrecision(task="multiclass", num_classes=self.num_levels, average="weighted"),
                        Accuracy(task="multiclass", num_classes=self.num_levels, average="weighted")
                    ]
                else:
                    metrics_list = [
                        MulticlassAUROC(num_classes=self.num_levels),
                        MulticlassAveragePrecision(num_classes=self.num_levels),
                        MulticlassF1Score(num_classes=self.num_levels, average="macro"),
                        MulticlassAccuracy(num_classes=self.num_levels),
                    ]
                if include_stats:
                    # We use average='micro' to get global TP/FP/etc counts across the levels for the specific label
                    metrics_list.append(MulticlassStatScores(num_classes=self.num_levels, average='micro'))
                return nn.ModuleList(metrics_list)

            # Validation: 4 metrics (No stats)
            self.val_metrics = nn.ModuleList([make_metrics(include_stats=False) for _ in range(self.num_labels)])
            # Test: 5 metrics (Includes stats)
            self.test_metrics = nn.ModuleList([make_metrics(include_stats=True) for _ in range(self.num_labels)])

        elif task == 'multiclass':
            if classification_head == 'linear':
                self.classification_head = nn.Linear(
                    d_model, num_classes
                )
            elif classification_head == 'mlp':
                self.classification_head = nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                logger.info("No valid classification head was chosen.\nValid options for classification head: linear, mlp")
                 
            self.val_metrics = MetricCollection({
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "auc": MulticlassAUROC(num_classes=num_classes),
                "auprc": MulticlassAveragePrecision(num_classes=num_classes),
            })

            self.test_metrics = MetricCollection({
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "auc": MulticlassAUROC(num_classes=num_classes),
                "auprc": MulticlassAveragePrecision(num_classes=num_classes),
                "stat": MulticlassStatScores(num_classes=num_classes, average="micro")
            })

        else:
            logger.info("No valid task was chosen.\nValid options for task: multiclass, multilabel")
        
        self.cls_token = cls_token

        self.criterion = nn.CrossEntropyLoss()

        self.metric_names = ["auc", "auprc", "f1", "acc", "stat"]

    def _evaluate_multilabel(self, batch, stage):
        x = batch.get("x")
        y = batch.get("y")

        x = x.permute(0, 2, 1)

        logits = self(x)
            
        if self.cls_token:
            cls_token = logits[:, 0, :]  # [batch, d_model]
        else:
            cls_token = torch.mean(logits, dim=1)  # [batch, d_model]

        # Pass through classification head 
        logits = self.classification_head(cls_token)
        logits = logits.view(
            logits.size(0),
            self.num_labels,
            self.num_levels
        )  # [B, num_labels, num_levels]
            
        loss = self.compute_loss(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)

        probs = F.softmax(logits, dim=-1)

        metrics = self.val_metrics if stage == "val" else self.test_metrics
        aggregated = {k: [] for k in self.metric_names}
            
        stat_aggregates = {"fpr": [], "fnr": [], "sens": [], "spec": []}

        for i in range(self.num_labels):
            preds_i = probs[:, i, :]
            targets_i = y[:, i]

            # Use zip to naturally stop when metrics run out (handles val vs test mismatch)
            for metric, name in zip(metrics[i], self.metric_names):
                metric.update(preds_i, targets_i)
                    
                if name == "stat":
                    # Logic for statistics (only runs if metric exists, i.e., in test)
                    stats_ = metric.compute().detach().cpu().numpy()
                    # Assuming average='micro' so we get [TP, FP, TN, FN, Sup]
                    tp = stats_[0]
                    fp = stats_[1]
                    tn = stats_[2]
                    fn = stats_[3]

                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                    self.log(f"{stage}_fpr_comp{i}", fpr, prog_bar=False)
                    self.log(f"{stage}_fnr_comp{i}", fnr, prog_bar=False)
                    self.log(f"{stage}_sens_comp{i}", sens, prog_bar=False)
                    self.log(f"{stage}_spec_comp{i}", spec, prog_bar=False)
                        
                    stat_aggregates["fpr"].append(fpr)
                    stat_aggregates["fnr"].append(fnr)
                    stat_aggregates["sens"].append(sens)
                    stat_aggregates["spec"].append(spec)
                    continue

                val = metric.compute()
                aggregated[name].append(val)
                self.log(f"{stage}_{name}_comp{i}", val, prog_bar=False)

        for name, vals in aggregated.items():
            # Fix: Check if vals is not empty before stacking
            if vals:
                self.log(
                    f"{stage}_{name}",
                    torch.stack(vals).mean(),
                    prog_bar=True
                )
        if stat_aggregates["sens"]: 
            for stat_name, values in stat_aggregates.items():
                avg_val = torch.tensor(values).mean()
                self.log(f"{stage}_avg_{stat_name}", avg_val, prog_bar=True)
    
    def _evaluate_multiclass(self, batch, stage):
        # Classification logic
        logits, y = self._shared_step(batch)
        loss = self.criterion(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
            
        metrics = self.val_metrics if stage == "val" else self.test_metrics

        for name, metric in metrics.items():
            if name != "stat":
                metric.update(logits, y)
                self.log(
                    f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True
                )
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


    def _shared_step(self, batch):
        x = batch.get("x") 
        y = batch.get("y") 
        x = x.permute(0, 2, 1)
        features = self.net(x) 
        if self.cls_token:
            cls_token = features[:, 0, :]  # [batch, d_model]
        else:
            cls_token = torch.mean(features, dim=1) 
        logits = self.classification_head(cls_token)
        return logits, y
    
    def compute_loss(self, logits, y):
        if self.task == 'multilabel':
            # logits: [B, num_labels, num_levels]
            # y:      [B, num_labels]
            loss = 0.0
            for i in range(self.num_labels):
                loss += F.cross_entropy(logits[:, i, :], y[:, i])
            return loss / self.num_labels
        else:
            return self.criterion(logits, y)
        
    def training_step(self, batch, batch_idx):
        if self.task == 'multilabel':
            logits, y = self._shared_step(batch)
            logits = logits.view(
                logits.size(0),
                self.num_labels,
                self.num_levels
            )  # [B, num_labels, num_levels]

            loss = self.compute_loss(logits, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss}
        else:
            # Classification logic
            logits, y = self._shared_step(batch)
            loss = self.criterion(logits, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def evaluate(self, batch, stage=None):
        if self.task == 'multilabel':
            self._evaluate_multilabel(batch, stage)
        else:
            self._evaluate_multiclass(batch, stage)
           
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        x = batch.get("x")
        y = batch.get("y")

        x = x.permute(0, 2, 1)

        logits = self(x)
        # Get the CLS token (first position)
        if self.cls_token:
            cls_token = logits[:, 0, :]  # [batch, d_model]
        else:
            cls_token = torch.mean(logits, dim=1)  # [batch, d_model]

        # Pass through classification head 
        logits = self.classification_head(cls_token)  # [batch, num_classes]

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
            "monitor": "val_loss",
        }
        
class SpectraEncoderRegressorLit(L.LightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            weight_decay: float = 1e-6,
            input_channels: int = 2,
            d_model: int = 64,
            encoder_depth: int = 1,
            nhead: int = 1,
            mlp_dim: int = 256,
            dim_head: int = 64,
            hidden_dim: int = 256,
            pos_embedding_type: str = 'sin_cos',
            dropout: float = 0.,
            emb_dropout: float = 0.,
            cls_token: bool = False,
            regression_head: str = 'linear',
        ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.net = spectra_encoder(
                input_dim=input_channels,
                dim=d_model,
                depth=encoder_depth,
                heads=nhead,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                pos_embedding_type=pos_embedding_type,
                dropout=dropout,
                emb_dropout=emb_dropout,
                )

        # Output 1 value for regression instead of num_classes
        if regression_head == 'linear':
            self.regression_head = nn.Linear(d_model, 1)
        elif regression_head == 'mlp':
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            logger.info("No valid regression head was chosen. Options: linear, mlp")
            
        self.cls_token = cls_token

        self.criterion = nn.MSELoss()

        metrics = MetricCollection({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
        })

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _shared_step(self, batch):
        x = batch.get("x") 
        y = batch.get("y") 
        x = x.permute(0, 2, 1)
        
        features = self.net(x) 
        if self.cls_token:
            cls_token = features[:, 0, :]  # [batch, d_model]
        else:
            cls_token = torch.mean(features, dim=1) 
            
        # [B, 1] -> [B] to match y's shape
        preds = self.regression_head(cls_token).squeeze(-1) 
        return preds, y
    
    def training_step(self, batch, batch_idx):
        preds, y = self._shared_step(batch)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluate(self, batch, stage="val"):
        preds, y = self._shared_step(batch)
        loss = self.criterion(preds, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
            
        metrics = self.val_metrics if stage == "val" else self.test_metrics
        
        metrics.update(preds, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
           
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        x = batch.get("x")
        y = batch.get("y")
        x = x.permute(0, 2, 1)

        features = self(x)
        if self.cls_token:
            cls_token = features[:, 0, :] 
        else:
            cls_token = torch.mean(features, dim=1) 

        # Return continuous predictions directly 
        preds = self.regression_head(cls_token).squeeze(-1)
        return preds, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }