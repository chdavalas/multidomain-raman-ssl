import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy, StatScores
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

from raman.mae.mae import MaskedAutoEncoderLit
import math

class LinearProbeLit(L.LightningModule):
    def __init__(
        self, 
        lr=1e-3, 
        weight_decay=0.00001, 
        num_classes=192, 
        pretrained_ckpt=None,
        cls_token=False, 
        freeze_enc=True, 
        input_channels=2,
        d_model=128,
        encoder_depth=4,
        decoder_depth=2,
        mlp_dim=512,
        nhead=4,
        dim_head=128,
        mask_ratio=0.4,
        pos_embedding_type='sin_cos',
        classification_head="linear", 
        hidden_dim: int = 256,
        mae_lr=0.00015,
        mae_weight_decay=0.000001,
        dropout=0.,
        emb_dropout=0.,
        use_fourier_pe=False,
        fourier_pe_dim=64,
        task="multiclass",
        weighted=False,
        regression=False 
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.freeze_enc = freeze_enc
        self.weight_decay = weight_decay
        self.cls_token = cls_token
        self.task = task
        self.regression = regression 

        if pretrained_ckpt is None:
            self.mae = MaskedAutoEncoderLit(
                input_channels=input_channels,
                d_model=d_model,
                encoder_depth=encoder_depth,
                decoder_depth=decoder_depth,
                mlp_dim=mlp_dim,
                nhead=nhead,
                dim_head=dim_head,
                mask_ratio=mask_ratio,
                pos_embedding_type=pos_embedding_type,
                lr=mae_lr,
                weight_decay=mae_weight_decay,
                dropout=dropout,
                emb_dropout=emb_dropout
            )
        else:
            self.mae = MaskedAutoEncoderLit.load_from_checkpoint(pretrained_ckpt)

        self.encoder = self.mae.model.encoder

        if self.freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.d_model = self.encoder.dim
        self.pos_embedding_type = self.encoder.pos_embedding_type
        
        # ---------------- HEAD SETUP ----------------
        if self.regression:
            out_dim = 1
        elif self.task == "multilabel":
            self.num_labels = 4 
            self.num_levels = num_classes 
            out_dim = self.num_labels * self.num_levels
        else:
            out_dim = num_classes

        if classification_head == "linear":
            self.classification_head = nn.Linear(self.d_model, out_dim)
        elif classification_head == "mlp":
            self.classification_head = nn.Sequential(
                nn.Linear(self.d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )

        # ---------------- LOSS SETUP ----------------
        if self.regression:
            self.criterion = nn.MSELoss()
            self.metrics_names = ["mse", "mae", "r2"]
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.metrics_names = ["auc", "f1", "auprc", "acc", "stat"]  

        # ---------------- METRICS SETUP ----------------
        if self.regression:
            self.val_metrics = nn.ModuleList([MeanSquaredError(), MeanAbsoluteError(), R2Score()])
            self.test_metrics = nn.ModuleList([MeanSquaredError(), MeanAbsoluteError(), R2Score()])
        elif self.task == "multilabel":
            def make_metrics():
                return nn.ModuleList([
                    AUROC(task="multiclass", num_classes=self.num_levels),
                    F1Score(task="multiclass", num_classes=self.num_levels, average="macro"),
                    AveragePrecision(task="multiclass", num_classes=self.num_levels),
                    Accuracy(task="multiclass", num_classes=self.num_levels),
                    StatScores(task="multiclass", num_classes=self.num_levels)
                ])
            self.val_metrics = nn.ModuleList([make_metrics() for _ in range(self.num_labels)])
            self.test_metrics = nn.ModuleList([make_metrics() for _ in range(self.num_labels)])
        else:
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
            if weighted:
                self.test_metrics = nn.ModuleList([
                    AUROC(task="multiclass", num_classes=num_classes, average="weighted"),
                    F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
                    AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted"),
                    Accuracy(task="multiclass", num_classes=num_classes, average="weighted")
            ])
            
        self.metrics = {
            "val": (self.val_metrics, self.metrics_names),
            "test": (self.test_metrics, self.metrics_names),
        }

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        
        if self.freeze_enc:
            with torch.no_grad():
                x = self.encoder(x)  
                if self.cls_token:
                    cls_token = x[:, 0, :]  
                else:
                    cls_token = torch.mean(x, dim=1)  
        else:
            x = self.encoder(x)  
            if self.cls_token:
                cls_token = x[:, 0, :]  
            else:
                cls_token = torch.mean(x, dim=1)  

        logits = self.classification_head(cls_token)  
        
        # Squeeze output dimension to [B] for regression
        if self.regression:
            logits = logits.squeeze(-1)
            
        return logits

    def compute_loss(self, logits, y):
        if self.task == 'multilabel':
            loss = 0.0
            for i in range(self.num_labels):
                loss += F.cross_entropy(logits[:, i, :], y[:, i])
            return loss / self.num_labels
        else:
            return self.criterion(logits, y)
    
    def training_step(self, batch, batch_idx):
        x = batch.get("x")
        y = batch.get("y")
        
        logits = self(x)
        
        if self.task == 'multilabel' and not self.regression:
            logits = logits.view(logits.size(0), self.num_labels, self.num_levels)

        loss = self.compute_loss(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        if self.regression:
            self._evaluate_regression(batch, stage)
        elif self.task == 'multilabel':
            self._evaluate_multilabel(batch, stage)
        else:
            self._evaluate_multiclass(batch, stage)

    def _evaluate_regression(self, batch, stage):
        x = batch.get("x")
        y = batch.get("y")
        
        preds = self(x)
        loss = self.compute_loss(preds, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        metrics, metrics_names = self.metrics[stage]
        for idx, metric in enumerate(metrics):
            name = metrics_names[idx]
            metric.update(preds, y)
            self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)

    def _evaluate_multiclass(self, batch, stage):
        x = batch.get("x")
        y = batch.get("y")
        
        logits = self(x)
        loss = self.compute_loss(logits, y)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        metrics, metrics_names = self.metrics[stage]
        argmax_requirement_metrics = ["f1", "acc"]

        for idx, metric in enumerate(metrics):
            name = metrics_names[idx]
            if name == "stat":
                metric.update(logits, y)
                stats_ = metric.compute().detach().cpu().numpy()
                
                if stats_.ndim == 2:
                    stats_ = stats_.sum(axis=0)
                    
                tp, fp, tn, fn = stats_[0], stats_[1], stats_[2], stats_[3]
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                self.log(f"{stage}_fpr", fpr, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"{stage}_fnr", fnr, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"{stage}_sens", sens, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"{stage}_spec", spec, on_step=False, on_epoch=True, prog_bar=True)
            else:
                if name in argmax_requirement_metrics:
                    probs = torch.argmax(logits, dim=1)
                    metric.update(probs, y)
                else:
                    metric.update(logits, y)
                self.log(
                    f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True
                )

    def _evaluate_multilabel(self, batch, stage):
        x = batch.get("x")
        y = batch.get("y")
        
        logits = self(x)
        logits = logits.view(logits.size(0), self.num_labels, self.num_levels)
        
        loss = self.compute_loss(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)

        probs = F.softmax(logits, dim=-1)
        metrics = self.val_metrics if stage == "val" else self.test_metrics
        
        aggregated = {k: [] for k in self.metrics_names if k != "stat"}
        aggregated.update({"fpr": [], "fnr": [], "sens": [], "spec": []})

        for i in range(self.num_labels):
            preds_i = probs[:, i, :]
            targets_i = y[:, i]

            for metric, name in zip(metrics[i], self.metrics_names):
                if name == "stat":
                    metric.update(preds_i, targets_i)
                    stats_ = metric.compute().detach().cpu().numpy()
                    
                    if stats_.ndim == 2:
                        stats_ = stats_.sum(axis=0)
                        
                    tp, fp, tn, fn = stats_[0], stats_[1], stats_[2], stats_[3]
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    self.log(f"{stage}_fpr_comp{i}", fpr, prog_bar=False)
                    self.log(f"{stage}_fnr_comp{i}", fnr, prog_bar=False)
                    self.log(f"{stage}_sens_comp{i}", sens, prog_bar=False)
                    self.log(f"{stage}_spec_comp{i}", spec, prog_bar=False)
                    
                    aggregated["fpr"].append(torch.tensor(fpr, dtype=torch.float32))
                    aggregated["fnr"].append(torch.tensor(fnr, dtype=torch.float32))
                    aggregated["sens"].append(torch.tensor(sens, dtype=torch.float32))
                    aggregated["spec"].append(torch.tensor(spec, dtype=torch.float32))
                else:
                    if name in ["f1", "acc"]:
                        preds_class_i = torch.argmax(preds_i, dim=-1)
                        metric.update(preds_class_i, targets_i)
                    else:
                        metric.update(preds_i, targets_i)
                    
                    val = metric.compute()
                    aggregated[name].append(val)
                    self.log(f"{stage}_{name}_comp{i}", val, prog_bar=False)

        for name, vals in aggregated.items():
            if vals:
                self.log(f"{stage}_{name}", torch.stack(vals).mean(), prog_bar=True)
                
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=50
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }