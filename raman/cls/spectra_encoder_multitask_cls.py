import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassAccuracy,
)
from raman.backbones.spectra_encoder import spectra_encoder
from raman.mae.mae import MaskedAutoEncoderLit
import logging

logger = logging.getLogger(__name__)


class SpectraEncoderMultiTaskClsLit(L.LightningModule):
    """
    Finetunes a (optionally pretrained) spectra_encoder jointly across multiple
    classification datasets/tasks, each routed through its own classification head
    since each dataset can have a different num_classes. Batches are expected to
    carry a "task" key (as produced by raman.dataloaders.multitask_dataloader.
    MultiTaskDataModule) naming which head/metrics to use.

    Does not subclass BaseClsLit: that base class assumes a single head/num_classes
    and can't express task-routed multi-head batches (same reason
    spectra_encoder_cls.SpectraEncoderClassifierLit doesn't subclass it either).
    """

    def __init__(
            self,
            tasks_config,
            pretrained_ckpt=None,
            freeze_encoder=False,
            input_channels=2,
            d_model=64,
            encoder_depth=1,
            nhead=1,
            mlp_dim=256,
            dim_head=64,
            pos_embedding_type='sin_cos',
            dropout=0.,
            emb_dropout=0.,
            cls_token=False,
            lr=1e-4,
            weight_decay=1e-6,
            encoder_lr=None,
        ):
        super().__init__()

        self.save_hyperparameters()

        self.tasks_config = tasks_config
        self.freeze_encoder = freeze_encoder
        self.cls_token = cls_token
        self.lr = lr
        self.weight_decay = weight_decay
        self.encoder_lr = encoder_lr

        if pretrained_ckpt is not None:
            mae_model = MaskedAutoEncoderLit.load_from_checkpoint(pretrained_ckpt)
            self.net = mae_model.model.encoder
        else:
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

        if self.freeze_encoder:
            for p in self.net.parameters():
                p.requires_grad = False

        self.d_model = self.net.dim

        self.heads = nn.ModuleDict()
        for cfg in tasks_config:
            task = cfg["task"]
            num_classes = cfg["num_classes"]
            head_type = cfg.get("classification_head", "linear")
            hidden_dim = cfg.get("hidden_dim", 256)
            if head_type == "linear":
                self.heads[task] = nn.Linear(self.d_model, num_classes)
            elif head_type == "mlp":
                self.heads[task] = nn.Sequential(
                    nn.Linear(self.d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
            else:
                raise ValueError(
                    f"No valid classification head was chosen for task '{task}'.\n"
                    "Valid options for classification_head: linear, mlp"
                )

        self.metrics = nn.ModuleDict()
        for cfg in tasks_config:
            task = cfg["task"]
            num_classes = cfg["num_classes"]
            for stage in ("val", "test"):
                self.metrics[f"{stage}_{task}"] = MetricCollection({
                    "acc": MulticlassAccuracy(num_classes=num_classes),
                    "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                    "auc": MulticlassAUROC(num_classes=num_classes),
                    "auprc": MulticlassAveragePrecision(num_classes=num_classes),
                })

        self.criterion = nn.CrossEntropyLoss()

    def _pool(self, features):
        if self.cls_token:
            return features[:, 0, :]  # [batch, d_model]
        return torch.mean(features, dim=1)  # [batch, d_model]

    def forward(self, x: torch.Tensor, task: str):
        if task not in self.heads:
            raise KeyError(
                f"Batch task '{task}' has no matching classification head. "
                f"Known tasks: {list(self.heads.keys())}. Check that tasks_config "
                "(model) and datasets_config (data) use the same task names."
            )
        x = x.permute(0, 2, 1)
        features = self.net(x)
        pooled = self._pool(features)
        return self.heads[task](pooled)

    def _shared_step(self, batch):
        task = batch.get("task")
        x = batch.get("x")
        y = batch.get("y")
        logits = self(x, task)
        loss = self.criterion(logits, y)
        return task, logits, y, loss

    def training_step(self, batch, batch_idx):
        task, logits, y, loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"train_loss_{task}", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def evaluate(self, batch, stage=None):
        task, logits, y, loss = self._shared_step(batch)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_loss_{task}", loss, on_step=False, on_epoch=True, prog_bar=False)

        metrics = self.metrics[f"{stage}_{task}"]
        metrics.update(logits, y)
        for name, metric in metrics.items():
            self.log(
                f"{stage}_{task}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True
            )

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        task = batch.get("task")
        x = batch.get("x")
        y = batch.get("y")

        logits = self(x, task)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return preds, y

    def configure_optimizers(self):
        if self.encoder_lr is not None and not self.freeze_encoder:
            param_groups = [
                {"params": self.net.parameters(), "lr": self.encoder_lr},
                {"params": self.heads.parameters(), "lr": self.lr},
            ]
        else:
            param_groups = self.parameters()

        optimizer = torch.optim.AdamW(
            param_groups,
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

    @classmethod
    def export_encoder(cls, ckpt_path, out_path):
        """
        Load a finetuned checkpoint and save its spectra_encoder's raw state_dict
        (unprefixed keys) as a standalone file, independent of the classification
        heads used to finetune it.

        Note: load_from_checkpoint reconstructs the module by re-running __init__
        from the checkpoint's saved hyperparameters, which (if pretrained_ckpt was
        set) re-triggers MaskedAutoEncoderLit.load_from_checkpoint against the
        *original* MAE checkpoint path baked into those hyperparameters -- that file
        must still be resolvable on disk, even though the actual finetuned weights
        loaded right after correctly overwrite it.
        """
        model = cls.load_from_checkpoint(ckpt_path)
        torch.save(model.net.state_dict(), out_path)


class SaveEncoderCallback(L.Callback):
    """
    Saves the LightningModule's `net` (the spectra_encoder backbone) as a
    standalone, raw state_dict once fit completes -- the same unprefixed-keys
    format savemodel.py produces, directly loadable via
    spectra_encoder(...).load_state_dict(torch.load(out_path)).
    """

    def __init__(self, out_path):
        self.out_path = out_path

    def on_fit_end(self, trainer, pl_module):
        torch.save(pl_module.net.state_dict(), self.out_path)
        logger.info(f"Saved finetuned encoder state_dict to {self.out_path}")
