import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics import CosineSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from raman.backbones.spectra_encoder import spectra_encoder
from torchmetrics import Metric
from raman.mae.mae import MaskedAutoEncoderLit

logger = logging.getLogger(__name__)

class SSIM1D(Metric):
    """
    Structural Similarity Index for 1D signals.
    Expects inputs of shape [B, T].
    """

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        window_size: int = 11,
        data_range: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        assert window_size % 2 == 1, "window_size must be odd"

        self.window_size = window_size
        self.data_range = data_range
        self.eps = eps

        # Accumulators
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        # Precompute Gaussian window (registered as buffer later)
        self.register_buffer("window", self._create_window(window_size))

    def _create_window(self, window_size):
        coords = torch.arange(window_size) - window_size // 2
        sigma = window_size / 6.0
        window = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = window / window.sum()
        return window.view(1, 1, -1)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds, target: [B, T]
        """

        preds = preds.unsqueeze(1)    # [B, 1, T]
        target = target.unsqueeze(1)  # [B, 1, T]

        window = self.window.to(preds.device)

        mu_x = F.conv1d(preds, window, padding=self.window_size // 2)
        mu_y = F.conv1d(target, window, padding=self.window_size // 2)

        sigma_x = F.conv1d(preds ** 2, window, padding=self.window_size // 2) - mu_x ** 2
        sigma_y = F.conv1d(target ** 2, window, padding=self.window_size // 2) - mu_y ** 2
        sigma_xy = (
            F.conv1d(preds * target, window, padding=self.window_size // 2)
            - mu_x * mu_y
        )

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = (
            (2 * mu_x * mu_y + C1)
            * (2 * sigma_xy + C2)
            / (
                (mu_x ** 2 + mu_y ** 2 + C1)
                * (sigma_x + sigma_y + C2)
                + self.eps
            )
        )

        self.ssim_sum += ssim_map.mean()
        self.n += 1

    def compute(self):
        return self.ssim_sum / self.n

class SpectraDenoiserLit(L.LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        weight_decay: float = 1e-6,
        input_channels: int = 2,  
        output_channels: int = 2, 
        d_model: int = 64,
        encoder_depth: int = 4,
        nhead: int = 8,
        mlp_dim: int = 256,
        dim_head: int = 64,
        pos_embedding_type: str = 'fourier',
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        cls_token: bool = False,
        pretrained_ckpt=None,
        freeze_encoder=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.cls_token = cls_token

        if pretrained_ckpt is not None:
            logger.info(f"Loading pretrained MAE encoder from {pretrained_ckpt}")
            mae_model = MaskedAutoEncoderLit.load_from_checkpoint(pretrained_ckpt)
            self.net = mae_model.model.encoder
            if freeze_encoder:
                logger.info("Freezing pretrained encoder parameters.")
                for p in self.net.parameters():
                    p.requires_grad = False
        else:
            logger.info("Using randomly initialized spectra encoder.")
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

        # self.output_head = nn.Linear(d_model, output_channels)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_channels)
        )

        self.criterion_train = nn.L1Loss()
        self.criterion_val = nn.MSELoss()

        self.metrics_names = ["mse", "mae", "psnr", "ssim_1d", "cosine_similarity", "r2"]

        def make_metrics():
            return nn.ModuleList([
                MeanSquaredError(),
                MeanAbsoluteError(),
                PeakSignalNoiseRatio(data_range=1.0),
                SSIM1D(window_size=11, data_range=1.0),
                CosineSimilarity(reduction='mean'),
                R2Score()
            ])
            
        self.train_metrics = make_metrics()
        self.val_metrics = make_metrics()
        self.test_metrics = make_metrics()

        self.metrics = {
            "train": (self.train_metrics, self.metrics_names),
            "val": (self.val_metrics, self.metrics_names),
            "test": (self.test_metrics, self.metrics_names),
        }

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch["x"] 
        y_true = batch["y"] 
        
        # [B, 2, T] -> [B, T, 2]
        x = x.permute(0, 2, 1)
        y_true = y_true.permute(0, 2, 1)

        features = self(x)

        # Remove CLS token (assuming it's at index 0)
        # Check if output is longer than input by 1
        if features.shape[1] == x.shape[1] + 1:
            features = features[:, 1:, :]

        y_hat = self.output_head(features)
        
        # Extract Intensities [B, T]
        preds = y_hat[:, :, 1]
        target = y_true[:, :, 1]
        
        # Loss for optimization (L1)
        loss = self.criterion_train(preds, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        metrics_list, names = self.metrics["train"]
        
        # Pre-calculate shapes needed for different metrics
        # Flat for regression (MSE, MAE, R2)
        preds_flat = preds.reshape(-1)
        target_flat = target.reshape(-1)
        
        # 4D for Image metrics (PSNR) -> [B, 1, 1, T]
        preds_img = preds.unsqueeze(1).unsqueeze(1)
        target_img = target.unsqueeze(1).unsqueeze(1)

        for idx, metric in enumerate(metrics_list):
            name = names[idx]
            
            if name == "psnr":
                metric.update(preds_img, target_img)
            elif name in {"cosine_similarity", "ssim_1d"}:
                metric.update(preds, target) # Expects [B, D]
            else:
                # MSE, MAE, R^2
                metric.update(preds_flat, target_flat)
                
            self.log(
                f"train_{name}", metric, 
                on_step=False, on_epoch=True, prog_bar=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"] 
        y_true = batch["y"] 
        
        # [B, 2, T] -> [B, T, 2]
        x = x.permute(0, 2, 1)
        y_true = y_true.permute(0, 2, 1)

        features = self(x)

        # Remove CLS token (assuming it's at index 0)
        # Check if output is longer than input by 1
        if features.shape[1] == x.shape[1] + 1:
            features = features[:, 1:, :]

        y_hat = self.output_head(features)

        preds = y_hat[:, :, 1]
        target = y_true[:, :, 1]
        
        loss_mse = F.mse_loss(preds, target)
        self.log("val_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True)
            
        metrics_list, names = self.metrics["val"]
        
        # Pre-calculate shapes needed for different metrics
        # Flat for regression (MSE, MAE, R2)
        preds_flat = preds.reshape(-1)
        target_flat = target.reshape(-1)
        
        # 4D for Image metrics (PSNR) -> [B, 1, 1, T]
        preds_img = preds.unsqueeze(1).unsqueeze(1)
        target_img = target.unsqueeze(1).unsqueeze(1)

        for idx, metric in enumerate(metrics_list):
            name = names[idx]
            
            if name == "psnr":
                metric.update(preds_img, target_img)
            elif name in {"cosine_similarity", "ssim_1d"}:
                metric.update(preds, target) # Expects [B, D]
            else:
                # MSE, MAE, R^2
                metric.update(preds_flat, target_flat)
                
            self.log(
                f"val_{name}", metric, 
                on_step=False, on_epoch=True, prog_bar=True
            )

    def test_step(self, batch, batch_idx):
        x = batch["x"] 
        y_true = batch["y"] 
        
        # [B, 2, T] -> [B, T, 2]
        x = x.permute(0, 2, 1)
        y_true = y_true.permute(0, 2, 1)

        features = self(x)

        # Remove CLS token (assuming it's at index 0)
        # Check if output is longer than input by 1
        if features.shape[1] == x.shape[1] + 1:
            features = features[:, 1:, :]

        y_hat = self.output_head(features)

        preds = y_hat[:, :, 1]
        target = y_true[:, :, 1]
        
        loss_mse = F.mse_loss(preds, target)
        self.log("test_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True)
        
        metrics_list, names = self.metrics["test"]
        
        # Pre-calculate shapes needed for different metrics
        # Flat for regression (MSE, MAE, R2)
        preds_flat = preds.reshape(-1)
        target_flat = target.reshape(-1)
        
        # 4D for Image metrics (PSNR) -> [B, 1, 1, T]
        preds_img = preds.unsqueeze(1).unsqueeze(1)
        target_img = target.unsqueeze(1).unsqueeze(1)

        for idx, metric in enumerate(metrics_list):
            name = names[idx]
            
            if name == "psnr":
                metric.update(preds_img, target_img)
            elif name in {"cosine_similarity", "ssim_1d"}:
                metric.update(preds, target) # Expects [B, D]
            else:
                # MSE, MAE, R^2
                metric.update(preds_flat, target_flat)
                
            self.log(
                f"test_{name}", metric, 
                on_step=False, on_epoch=True, prog_bar=True
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }