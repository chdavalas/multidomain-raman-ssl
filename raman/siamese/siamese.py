import lightning as L
import torch
import torch.nn as nn
from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy, StatScores
from raman.backbones.spectra_encoder import spectra_encoder
import torch.nn.functional as F
import logging
from raman.backbones.siamese import SiameseNet
from torchmetrics import MetricCollection
import numpy as np
from torchmetrics import Accuracy, F1Score, Precision, Recall
import sys
from sklearn.linear_model import enet_path
from raman.mae.mae import MaskedAutoEncoderLit
from collections import defaultdict

from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = label * 0.5 * torch.pow(distance, 2) + \
               (1 - label) * 0.5 * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()

class SiameseLit(L.LightningModule):
    def __init__(
            self,
            encoder_type="spectra",
            num_classes: int = 2,
            lr: float = 1e-4,
            weight_decay: float = 1e-6,
            input_channels: int = 2,
            d_model: int = 64,
            encoder_depth: int = 1,
            nhead: int = 1,
            mlp_dim: int = 256,
            dim_head: int = 64,
            pos_embedding_type: str = 'sin_cos',
            dropout: float = 0.,
            emb_dropout: float = 0.,
            cls_token: bool = False,
            projection_dim: int = 64,
            closs_margin: float = 1.0,
            shared_weights: bool = False,
            classification: bool = True,
            loss_ratio: float = 0.5,
            pretrained_ckpt=None, 
            freeze_encoder=False,
            weighted=False,
            classification_head: str = 'mlp',
            regression: bool = False,
            per_spirit_metrics: bool = False
            ):
        super().__init__()

        self.save_hyperparameters()
        self.regression = regression
        self.per_spirit_metrics = per_spirit_metrics
        self.num_classes = num_classes 
        self.lr = lr
        self.weight_decay = weight_decay
        self.shared_weights = shared_weights
        self.classification = classification
        self.loss_ratio = loss_ratio
        self.encoder_output_dim = d_model
        self.cls_token = cls_token

        # Output Dimension Logic
        out_dim = 1 if self.regression else self.num_classes

        if encoder_type == 'spectra':
            if pretrained_ckpt is not None:
                logger.info(f"Loading pretrained MAE encoder from {pretrained_ckpt}")
                mae_model = MaskedAutoEncoderLit.load_from_checkpoint(pretrained_ckpt)
                self.encoder = mae_model.model.encoder
                if freeze_encoder:
                    logger.info("Freezing pretrained encoder parameters.")
                    for p in self.encoder.parameters():
                        p.requires_grad = False
            else:
                logger.info("Using randomly initialized spectra encoder.")
                self.encoder = spectra_encoder(
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
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        if self.shared_weights:
            self.net = SiameseNet(
                self.encoder,
                projection_dim=projection_dim,
                concat_classify=self.classification,
                cls_token=self.cls_token,
                out_dim=out_dim, 
                head_type=classification_head
            )
        else:
            self.net = SiameseNet(
                self.encoder,
                self.encoder,
                projection_dim=projection_dim,
                concat_classify=self.classification,
                cls_token=self.cls_token,
                out_dim=out_dim,
                head_type=classification_head
            )

        self.contrastive_loss = ContrastiveLoss(margin=closs_margin)

        # ---------------- LOSS SETUP ----------------
        if self.regression:
            self.classification_loss = nn.MSELoss()
            self.metrics_names = ["mse", "mae", "r2"]
        else:
            self.classification_loss = nn.CrossEntropyLoss()
            self.metrics_names = ["auc", "f1", "auprc", "acc", "stat"]

        # ---------------- METRICS SETUP ----------------
        if self.regression:
            self.val_metrics = nn.ModuleList([MeanSquaredError(), MeanAbsoluteError(), R2Score()])
            self.test_metrics = nn.ModuleList([MeanSquaredError(), MeanAbsoluteError(), R2Score()])
        else:
            self.val_metrics = nn.ModuleList([
                AUROC(task="multiclass", num_classes=self.num_classes),
                F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
                AveragePrecision(task="multiclass", num_classes=self.num_classes),
                Accuracy(task="multiclass", num_classes=self.num_classes),
                StatScores(task="multiclass", num_classes=self.num_classes)
            ])
            self.test_metrics = nn.ModuleList([
                AUROC(task="multiclass", num_classes=self.num_classes),
                F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
                AveragePrecision(task="multiclass", num_classes=self.num_classes),
                Accuracy(task="multiclass", num_classes=self.num_classes),
                StatScores(task="multiclass", num_classes=self.num_classes)
            ])
            if weighted:
                self.test_metrics = nn.ModuleList([
                    AUROC(task="multiclass", num_classes=num_classes, average="weighted"),
                    F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
                    AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted"),
                    Accuracy(task="multiclass", num_classes=num_classes, average="weighted"),
                    StatScores(task="multiclass", num_classes=self.num_classes, average="weighted")
            ])
            
        self.metrics = {
            "val": (self.val_metrics, self.metrics_names),
            "test": (self.test_metrics, self.metrics_names),
        }

    def forward(self, x1, x2):
        return self.net(x1, x2)

    def training_step(self, batch, batch_idx):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]
        
        # Don't cast to .long() if doing regression
        if not self.regression:
            y = y.long()

        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        enc1, enc2, logits = self(x1, x2)
        
        if self.regression:
            logits = logits.squeeze(-1)

        l = self.loss_ratio
        loss = (1 - l) * self.classification_loss(logits, y)
        loss += l * self.contrastive_loss(enc1, enc2, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss                                          

    def evaluate(self, batch, stage):
        x1 = batch["x"]
        x2 = batch["ref"]
        y = batch["y"]
        
        if not self.regression:
            y = y.long()

        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        enc1, enc2, logits = self(x1, x2)
        
        if self.regression:
            logits = logits.squeeze(-1)

        l = self.loss_ratio
        loss = (1 - l) * self.classification_loss(logits, y)
        loss += l * self.contrastive_loss(enc1, enc2, y)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

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

    # ==========================================
    # Per-Spirit Testing Logic
    # ==========================================
    
    def on_test_epoch_start(self):
        # Only initialize if the flag is True
        if self.per_spirit_metrics:
            self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        # Run standard global evaluate
        self.evaluate(batch, "test")
        
        # Accumulate outputs ONLY if flag is True and spirit info exists
        if self.per_spirit_metrics and "spirit" in batch:
            x1 = batch["x"]
            x2 = batch["ref"]
            y = batch["y"]
        
            if not self.regression:
                y = y.long()
            
            x1 = x1.permute(0, 2, 1)
            x2 = x2.permute(0, 2, 1)
            
            
            enc1, enc2, logits = self(x1, x2)
            if self.regression:
                logits = logits.squeeze(-1)
                            
            self.test_step_outputs.append({
                "logits": logits.detach(),
                "y": y.detach(),
                "spirits": batch["spirit"]
            })
    
    def on_test_epoch_end(self):
        # Early exit if the flag is False or nothing was accumulated
        if not getattr(self, "per_spirit_metrics", False) or not getattr(self, "test_step_outputs", None):
            return
            
        # Group logits and targets by spirit
        from collections import defaultdict
        spirit_logits = defaultdict(list)
        spirit_targets = defaultdict(list)
        
        for out in self.test_step_outputs:
            logits = out["logits"]
            targets = out["y"]
            spirits = out["spirits"]
            
            for i, sp in enumerate(spirits):
                spirit_logits[sp].append(logits[i])
                spirit_targets[sp].append(targets[i])
                
        # Calculate individual metrics per spirit
        for sp in spirit_logits:
            sp_logits_tensor = torch.stack(spirit_logits[sp]).to(self.device)
            sp_targets_tensor = torch.stack(spirit_targets[sp]).to(self.device)
            
            # CALCULATE GLOBAL METRICS (Including Global R2)
            for idx, metric in enumerate(self.test_metrics):
                name = self.metrics_names[idx]
                
                # Clone metric for isolated state
                local_metric = metric.clone().to(self.device)
                local_metric.update(sp_logits_tensor, sp_targets_tensor)
                
                if self.regression:
                    val = local_metric.compute()
                    self.log(f"test_{name}_{sp}", val)
                else:
                    if name != "stat":
                        val = local_metric.compute()
                        self.log(f"test_{name}_{sp}", val)
                    else:
                        stats_ = local_metric.compute().detach().cpu().numpy()
                        tp, fp, tn, fn = stats_[0], stats_[1], stats_[2], stats_[3]
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        self.log(f"test_fpr_{sp}", fpr)
                        self.log(f"test_fnr_{sp}", fnr)
                        self.log(f"test_sens_{sp}", sens)
                        self.log(f"test_spec_{sp}", spec)
            
            # CALCULATE SPECIFIC LOD MAE (Just for 1.0 mg/ml samples)
            if self.regression:
                # Find all samples where the true target is 1.0 (using isclose for float safety)
                lod_target_val = 1.0
                mask_1_0 = torch.isclose(sp_targets_tensor, torch.tensor(lod_target_val).to(self.device), atol=1e-3)                
                
                if mask_1_0.sum() > 0:
                    lod_preds = sp_logits_tensor[mask_1_0]
                    lod_targets = sp_targets_tensor[mask_1_0]
                    
                    # Calculate Mean Absolute Error manually for just these samples
                    lod_mae = torch.nn.functional.l1_loss(lod_preds, lod_targets)
                    self.log(f"test_LOD_MAE_1.0_{sp}", lod_mae)
                else:
                    logger.warning(f"No 1.0 mg/ml samples found in test set for {sp}")
           
            # -------------------------------------------------------------
            # CALCULATE IUPAC ANALYTICAL LOD
            # -------------------------------------------------------------
            if self.regression:
                # Identify the Blank Samples (Assuming your blank label is 0.0)
                blank_val = 0.0 
                blank_mask = torch.isclose(sp_targets_tensor, torch.tensor(blank_val).to(self.device), atol=1e-3)
                
                # We need at least 2 blank samples to calculate a valid standard deviation
                if blank_mask.sum() > 1:
                    blank_preds = sp_logits_tensor[blank_mask]
                    # Calculate standard deviation (sigma) of the blank predictions
                    sigma_blanks = torch.std(blank_preds, unbiased=True)
                    
                    # Calculate the Slope of the Calibration Curve
                    x = sp_targets_tensor  # True Concentrations
                    y = sp_logits_tensor   # Predicted Concentrations
                    
                    x_mean = x.mean()
                    y_mean = y.mean()
                    
                    # Denominator of the slope formula (Variance of X)
                    x_var_sum = torch.sum((x - x_mean) ** 2)
                    
                    if x_var_sum > 0:
                        # Slope = Cov(X,Y) / Var(X)
                        slope = torch.sum((x - x_mean) * (y - y_mean)) / x_var_sum
                        
                        # Compute the IUPAC LOD
                        if slope > 0:
                            iupac_lod = 3.3 * (sigma_blanks / slope)
                            self.log(f"test_IUPAC_LOD_{sp}", iupac_lod)
                        else:
                            logger.warning(f"Negative/zero slope ({slope:.4f}) for {sp}, cannot calculate IUPAC LOD.")
                    else:
                        logger.warning(f"No variance in true targets for {sp}, cannot calculate slope.")
                else:
                    logger.warning(f"Not enough blank ({blank_val} mg/ml) samples found in test set for {sp} to calculate IUPAC LOD.")
                        
        self.test_step_outputs.clear() # Clear memory

    def predict_step(self, batch, batch_idx):
        x1 = batch["x"].permute(0, 2, 1)
        x2 = batch["ref"].permute(0, 2, 1)
        
        enc1, enc2, logits = self(x1, x2)
        
        output = {
            "enc1": enc1,
            "enc2": enc2,
            "logits": logits,
        }
        
        # Regression predictions vs Classification predictions
        if self.regression:
            logits = logits.squeeze(-1)
            output["preds"] = logits
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            output["probs"] = probs
            output["preds"] = preds
        
        # Include any available metadata
        for key in ["y", "component_name", "mixture_index"]:
            if key in batch:
                output[key] = batch[key]
            
        return output                            

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }