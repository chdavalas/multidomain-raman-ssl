import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC, AveragePrecision, F1Score, Accuracy
from raman.mae.mae import MaskedAutoEncoderLit

class SiameseCrossAttentionProbeLit(L.LightningModule):
    def __init__(
        self,
        pretrained_ckpt: str,
        num_classes: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        nhead: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = True
        weighted=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # 1. Load MAE and Freeze
        mae_model = MaskedAutoEncoderLit.load_from_checkpoint(pretrained_ckpt)
        self.encoder = mae_model.model.encoder
        self.d_model = self.encoder.dim
        
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 2. Cross-Attention Mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 3. Classification Head
        self.classification_head = nn.Linear(self.d_model, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # 4. Metrics
        self.metrics_names = ["auc", "f1", "auprc", "acc"]
        def make_metrics(weighted=False):
            if weighted:
                return nn.ModuleList([
                    AUROC(task="multiclass", num_classes=num_classes, average="weighted"),
                    F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
                    AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted"),
                    Accuracy(task="multiclass", num_classes=num_classes, average="weighted")
                ])
            return nn.ModuleList([
                AUROC(task="multiclass", num_classes=num_classes),
                F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                AveragePrecision(task="multiclass", num_classes=num_classes),
                Accuracy(task="multiclass", num_classes=num_classes)
            ])
        self.val_metrics = make_metrics(weighted=weighted)
        self.test_metrics = make_metrics(weighted=weighted)

    def forward(self, x_mix, x_ref, return_attention=False):
        # Ensure correct shape [B, T, C]
        x_mix = x_mix.permute(0, 2, 1)
        x_ref = x_ref.permute(0, 2, 1)

        # Encode both with frozen encoder
        with torch.no_grad():
            memory_mix = self.encoder(x_mix) # Shape: [B, Seq_Len, d_model]
            memory_ref = self.encoder(x_ref) 

        # We want to see how the Reference (Query) attends to the Mixture (Key/Value)
        # To get a single representation, we can just use the CLS token of the Reference as the Query
        query = memory_ref[:, 0:1, :] # Shape: [B, 1, d_model]

        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=memory_mix,
            value=memory_mix,
            need_weights=True,
            average_attn_weights=False
        )

        attn_output = attn_output.squeeze(1) # Shape: [B, d_model]
        logits = self.classification_head(attn_output)

        if return_attention:
            return logits, attn_weights

        return logits

    def training_step(self, batch, batch_idx):
        x_mix = batch["x"]
        x_ref = batch["ref"]
        y = batch["y"].long()

        logits = self(x_mix, x_ref)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _shared_eval_step(self, batch, stage):
        x_mix = batch["x"]
        x_ref = batch["ref"]
        y = batch["y"].long()

        logits = self(x_mix, x_ref)
        loss = self.criterion(logits, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        metrics = self.val_metrics if stage == "val" else self.test_metrics
        for metric, name in zip(metrics, self.metrics_names):
            metric.update(logits, y)
            self.log(f"{stage}_{name}", metric, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}