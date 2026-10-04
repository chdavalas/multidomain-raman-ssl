#######################
# Concentration Levels#
#######################

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy
from raman.backbones.ssp import SSPModel
import logging
import numpy as np
from sklearn.linear_model import enet_path
from torchmetrics import MetricCollection

logger = logging.getLogger(__name__)

class SSPLit(L.LightningModule):
    def __init__(
        self,
        num_classes=4,
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
        test_threshold=None,
        external_eval: bool = True, 
        pretrained_ckpt=None, 
        freeze_encoder=False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SSPModel(
            num_classes=num_classes,
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
        self.external_eval = external_eval
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_threshold = test_threshold
        self.num_classes = num_classes

        # CrossEntropyLoss for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics for multi-class classification
        self.metrics_names = ["acc", "f1", "auc", "auprc"]
        
        # Use multi-class metrics - note: AUROC and AveragePrecision need probabilities, not predictions
        # For multi-class, we need separate metrics for different input types
        self.val_metrics_acc_f1 = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        })
        
        self.val_metrics_auc_auprc = MetricCollection({
            "auc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            "auprc": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro")
        })
        
        # Clone for test
        self.test_metrics_acc_f1 = self.val_metrics_acc_f1.clone()
        self.test_metrics_auc_auprc = self.val_metrics_auc_auprc.clone()
        
        # For external evaluation mode
        self.test_step_outputs = []
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]
        
        logits = self(x1, x2)
        loss = self.criterion(logits, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Also log training accuracy
        preds = torch.argmax(logits, dim=1)
        train_acc = (preds == y).float().mean()
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def evaluate(self, batch, stage=None):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]
        
        logits = self(x1, x2)
        loss = self.criterion(logits, y)
        
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        if stage == "val":
            # Update accuracy and F1 metrics with predictions
            self.val_metrics_acc_f1.update(preds, y)
            
            # Update AUC and AUPRC metrics with probabilities
            self.val_metrics_auc_auprc.update(probs, y)
            
            # Log all metrics
            metrics_acc_f1 = self.val_metrics_acc_f1.compute()
            metrics_auc_auprc = self.val_metrics_auc_auprc.compute()
            
            for name, metric in {**metrics_acc_f1, **metrics_auc_auprc}.items():
                self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True)
                
            # Reset after logging
            self.val_metrics_acc_f1.reset()
            self.val_metrics_auc_auprc.reset()
            
        else:
            # For test in supervised mode
            self.test_metrics_acc_f1.update(preds, y)
            self.test_metrics_auc_auprc.update(probs, y)
 
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")   

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]

        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        _, _, logits = self(x1, x2)

        probs = torch.softmax(logits, dim=1)                
        preds = torch.argmax(probs, dim=1)

        return preds, probs, y     
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }