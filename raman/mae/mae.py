import lightning as L
import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, SymmetricMeanAbsolutePercentageError, ExplainedVariance
from torchmetrics import CosineSimilarity, Metric
from torchmetrics.image import PeakSignalNoiseRatio
from torch.optim import AdamW
import logging
from torch.nn import functional as F
from raman.backbones.mae import MaskedAutoencoder

logger = logging.getLogger(__name__)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil


class SSIM1D(Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(self, window_size: int = 11, data_range: float = 1.0, eps: float = 1e-8):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.data_range = data_range
        self.eps = eps
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.register_buffer("window", self._create_window(window_size))

    def _create_window(self, window_size):
        coords = torch.arange(window_size) - window_size // 2
        sigma = window_size / 6.0
        window = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = window / window.sum()
        return window.view(1, 1, -1)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)
        window = self.window.to(preds.device)
        mu_x = F.conv1d(preds, window, padding=self.window_size // 2)
        mu_y = F.conv1d(target, window, padding=self.window_size // 2)
        sigma_x = F.conv1d(preds ** 2, window, padding=self.window_size // 2) - mu_x ** 2
        sigma_y = F.conv1d(target ** 2, window, padding=self.window_size // 2) - mu_y ** 2
        sigma_xy = (F.conv1d(preds * target, window, padding=self.window_size // 2) - mu_x * mu_y)
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / 
                    ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + self.eps))
        self.ssim_sum += ssim_map.mean()
        self.n += 1

    def compute(self):
        return self.ssim_sum / self.n

def shade_masked_regions(ax, wavenumbers, mask_indices, **kwargs):
    """
    Shade contiguous masked regions on a spectrum plot.

    Parameters
    ----------
    ax:
        Matplotlib axis.
    wavenumbers:
        1D tensor or array of shape [T].
    mask_indices:
        1D tensor or array containing masked sample indices.
    """
    wavenumbers = torch.as_tensor(wavenumbers).detach().cpu()
    mask_indices = torch.as_tensor(mask_indices).detach().cpu().long()

    if mask_indices.numel() == 0:
        return

    mask_indices = torch.unique(mask_indices, sorted=True)

    # Split indices whenever consecutive values differ by more than one.
    split_points = torch.where(torch.diff(mask_indices) > 1)[0] + 1
    regions = torch.tensor_split(mask_indices, split_points.tolist())

    for region_number, region in enumerate(regions):
        start_idx = region[0].item()
        end_idx = region[-1].item()

        # Extend the shaded region by half a sampling interval so that
        # the complete masked observations are visually covered.
        if start_idx > 0:
            left = (
                wavenumbers[start_idx - 1] + wavenumbers[start_idx]
            ).item() / 2
        else:
            left = wavenumbers[start_idx].item()

        if end_idx < len(wavenumbers) - 1:
            right = (
                wavenumbers[end_idx] + wavenumbers[end_idx + 1]
            ).item() / 2
        else:
            right = wavenumbers[end_idx].item()

        ax.axvspan(
            left,
            right,
            alpha=0.2,
            label="Masked region" if region_number == 0 else None,
            **kwargs,
        )

class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.01, lambda_param=0.2, penalty_weight=0.01):
        super().__init__()
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.penalty_weight = penalty_weight

    def forward(self, recon, recon_voigt, peaks, recon_denoised, target, task):

        weights = F.softmax(target / self.temperature, dim=1)

        N = target.shape[1]
        weights = weights * N
        weights = weights.detach() 
        A_penalty = peaks[-1].abs().mean()

        w_mae, w_voigt, w_denoise = 1.0, 1.0, 1.0

        full_spectra_loss = F.mse_loss(recon, target)
        denoising_loss = F.l1_loss(recon_denoised, target)
        peak_loss = torch.mean(weights * (recon_voigt - target) ** 2)
        full_voigt_spectra_loss = F.mse_loss(recon_voigt, target) 
    
        if task == "denoising":
            w_mae = 0.0; w_voigt = 0.0
        elif task == "voigt":
            w_mae = 0.0; w_denoise = 0.0
        else:
            w_voigt = 0.0; w_denoise = 0.0

        loss = (
              w_mae * full_spectra_loss
            + w_voigt * (
                self.lambda_param * peak_loss 
                + (1 - self.lambda_param) * full_voigt_spectra_loss 
                + self.penalty_weight * A_penalty
            )
            + w_denoise * denoising_loss
        )

        return loss


class MaskedAutoEncoderLit(L.LightningModule):
    def __init__(
        self, 
        input_channels=2, 
        d_model=64, 
        encoder_depth=1, 
        decoder_depth=1, 
        nhead=1,
        mlp_dim=256,
        dim_head=64,
        mask_ratio=0.1,
        pos_embedding_type='sin_cos',
        dropout=0., 
        emb_dropout=0.,
        use_fourier_pe=False,
        fourier_pe_dim=64,
        lr=1e-4, 
        weight_decay=0.000001,
        wv_in_loss=True
        ):
        super().__init__()

        self.save_hyperparameters()
        self.wv_in_loss = wv_in_loss
        self.model = MaskedAutoencoder(
            input_channels=input_channels, 
            d_model=d_model, 
            encoder_depth=encoder_depth, 
            decoder_depth=decoder_depth, 
            nhead=nhead, 
            mlp_dim=mlp_dim,
            dim_head=dim_head, 
            mask_ratio=mask_ratio, 
            dropout=dropout, 
            emb_dropout=emb_dropout, 
            pos_embedding_type=pos_embedding_type,
            use_fourier_pe=use_fourier_pe,
            fourier_pe_dim=fourier_pe_dim
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.criterion = CombinedLoss(temperature=0.01, lambda_param=0.2, penalty_weight=0.01)
        def make_pretraining_metrics():
            return nn.ModuleDict({
               "mse" : MeanSquaredError(),
               "mae" : MeanAbsoluteError(),
               "r2" : R2Score(),
               "smape" : SymmetricMeanAbsolutePercentageError(),
               "explained_variance" : ExplainedVariance(),
            })

        def make_denoising_metrics():
            return nn.ModuleDict({
                "mse": MeanSquaredError(),
                "r2": R2Score(),
                "psnr": PeakSignalNoiseRatio(data_range=1.0),
                "ssim_1d": SSIM1D(window_size=11, data_range=1.0),
                "cosine_similarity": CosineSimilarity(reduction='mean')
            })

        self.metrics = nn.ModuleDict({
            "train_pretraining": make_pretraining_metrics(),
            "train_denoising": make_denoising_metrics(),
            
            "val_pretraining": make_pretraining_metrics(),
            "val_denoising": make_denoising_metrics(),
            
            "test_pretraining": make_pretraining_metrics(),
            "test_denoising": make_denoising_metrics(),
        })
        
    def forward(self, x: torch.Tensor, task: str = None):
        return self.model(x, task)
    
    def compute_loss(self, batch):
        x = batch.get("x")
        dataset_task = batch.get("task")
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)

        model_output = self(x, dataset_task)

        reconstructed = model_output["reconstruction"]
        reconstructed_unmasked = reconstructed.clone()

        reconstructed_voigt = model_output["reconstruction_voigt"]
        reconstructed_denoised = model_output["reconstruction_denoised"]

        mask_indices = model_output["mask_idx"]
        mask_task = model_output["task"]
        voigt_peaks = model_output["voigt_peaks"]

        # Resolve actual task
        if mask_task == "voigt_peak" and dataset_task == "pretraining":
            actual_task = "voigt"
        elif mask_task != "voigt_peak" and dataset_task == "pretraining":
            actual_task = mask_task
        elif dataset_task == "denoising": # for denoising the actual task is the dataset task
            actual_task = dataset_task

        # Targets & predictions
        if actual_task == 'denoising':
            target = batch.get("y").permute(0, 2, 1) 
            masked_target = target.clone() # Not really masked, but keep the same shape with for consistency

        else:  # pretraining
            target = x.clone()
            masked_target = x.gather(1, mask_indices.unsqueeze(2).expand(-1, -1, x.size(2)))
        
        if mask_indices is not None:
            reconstructed = reconstructed.gather(1, mask_indices.unsqueeze(2).expand(-1, -1, reconstructed.size(2)))
            reconstructed_denoised = reconstructed_denoised.gather(1, mask_indices.unsqueeze(2).expand(-1, -1, reconstructed_denoised.size(2)))

        if not self.wv_in_loss:
            reconstructed = reconstructed[:, :, 1].unsqueeze(2)
            reconstructed_voigt = reconstructed_voigt[:, :, 1].unsqueeze(2)
            reconstructed_denoised = reconstructed_denoised[:, :, 1].unsqueeze(2)
            masked_target = masked_target[:, :, 1].unsqueeze(2)

        loss = self.criterion(
            reconstructed,
            reconstructed_voigt,
            voigt_peaks,
            reconstructed_denoised,
            masked_target,
            actual_task
        )

        if actual_task == "denoising":
            preds = reconstructed_denoised
        else:
            preds = reconstructed_unmasked

        return {
            "loss": loss,
            "actual_task": actual_task,
            "batch_size": batch_size,
            "preds": preds,
            "target": target,
            "unmasked_reconstructed": reconstructed_unmasked,
            "reconstructed": reconstructed,
            "reconstructed_voigt": reconstructed_voigt,
            "reconstructed_denoised": reconstructed_denoised,
            "voigt_peaks": voigt_peaks,
            "mask_task": mask_task,
            "mask_idx": mask_indices
        }

    def compute_and_log_metrics(self, stage, actual_task, preds, target, batch_size):

        preds_flat = preds.reshape(-1)
        target_flat = target.reshape(-1)

        preds_img = preds.reshape(batch_size, 1, 1, -1)
        target_img = target.reshape(batch_size, 1, 1, -1)

        preds_seq = preds.reshape(batch_size, -1)
        target_seq = target.reshape(batch_size, -1)

        general_task = "denoising" if actual_task == "denoising" else "pretraining"

        metric_dict = self.metrics[f"{stage}_{general_task}"]

        for name, metric in metric_dict.items():
            if name in ["mse", "mae", "r2", "smape", "explained_variance"]:
                metric.update(preds_flat, target_flat)
            elif name == "psnr":
                metric.update(preds_img, target_img)
            elif name in ["ssim_1d", "cosine_similarity"]:
                metric.update(preds_seq, target_seq)

            self.log(
                f"{stage}_{general_task}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size
            )
    def training_step(self, batch, batch_idx):

        out = self.compute_loss(batch)

        self.log("train_loss", out["loss"], on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True, batch_size=out["batch_size"])

        general_task = "denoising" if out["actual_task"] == "denoising" else "pretraining"

        self.log(f"train_{general_task}_loss", out["loss"],
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=out["batch_size"])

        self.compute_and_log_metrics(
            "train",
            out["actual_task"],
            out["preds"],
            out["target"],
            out["batch_size"]
        )

        return {"loss": out["loss"]}

    # Helper function to print out ddp unused parameters (for debugging)
    # def on_after_backward(self):
    #     print("on_after_backward enter")
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)
    #     print("on_after_backward exit")

    def validation_step(self, batch, batch_idx):
        out = self.compute_loss(batch)

        self.log("val_loss", out["loss"], on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True, batch_size=out["batch_size"])

        general_task = "denoising" if out["actual_task"] == "denoising" else "pretraining"

        self.log(f"val_{general_task}_loss", out["loss"],
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=out["batch_size"])

        self.compute_and_log_metrics(
            "val",
            out["actual_task"],
            out["preds"],
            out["target"],
            out["batch_size"]
        )

        return {"val_loss": out["loss"]}

    def test_step(self, batch, batch_idx):
        out = self.compute_loss(batch)
        x = batch.get("x")
        mask_idx = out["mask_idx"]

        x = x.permute(0, 2, 1)

        self.log("test_loss", out["loss"], on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True, batch_size=out["batch_size"])

        general_task = "denoising" if out["actual_task"] == "denoising" else "pretraining"      

        self.log(f"test_{general_task}_loss", out["loss"],
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=out["batch_size"])

        self.compute_and_log_metrics(
            "test",
            out["actual_task"],
            out["preds"],
            out["target"],
            out["batch_size"]
        )

        # --- PLOTTING (unchanged logic) ---
        if batch_idx < 100:
            base_dir = self.logger.log_dir if self.logger and self.logger.log_dir else "."
            i = torch.randperm(4)[0].item()

            wv = x.detach().cpu()[i, :, 0]

            fig = plt.figure(figsize=(9, 4))
            fig.subplots_adjust(bottom=0.35)
            ax = plt.gca()

            if out["actual_task"] == "denoising":
                ax.set_title(f"Sample {batch_idx} - Denoising")
                ax.plot(wv, x.detach().cpu()[i, :, 1], label="Noisy Input", lw=1.5, alpha=0.5)
                ax.plot(wv, out["target"].detach().cpu()[i, :, 1], label="Clean Target", lw=1.5)
                ax.plot(wv, out["reconstructed_denoised"].detach().cpu()[i, :, 1], label="Denoised", lw=1.5)
            else:
                sample_mask_idx = mask_idx[i].detach().cpu().long()
                masked_wv = wv[sample_mask_idx]

                ax.axvspan(
                    masked_wv.min().item(),
                    masked_wv.max().item(),
                    alpha=0.2,
                    label="Masked region",
                )
                ax.set_title(f"Sample {batch_idx} - {out['actual_task'].capitalize()}")
                ax.plot(wv, x.detach().cpu()[i, :, 1], label="Input", lw=1.5, alpha=0.5)
                if out["mask_task"] == "voigt_peak":
                    ax.plot(wv.gather(0, mask_idx[i].detach().cpu()), out["reconstructed_voigt"].detach().cpu()[i, :, 1], label="Voigt Recon", lw=1.5)
                else:
                    ax.plot(wv, out["unmasked_reconstructed"].cpu()[i, :, 1], label="Recon", lw=1.5)

            ax.set_xlabel("Wavenumber")
            ax.set_ylabel("Intensity")
            ax.legend()

            plt.savefig(os.path.join(base_dir, f"sample_{batch_idx}_{out['actual_task']}.png"))
            plt.close(fig)

        return {
            "reconstruction": out["reconstructed"].cpu().detach(),
            "reconstruction_voigt": out["reconstructed_voigt"].cpu().detach(),
            "task": out["mask_task"],
            "voigt_peaks": out["voigt_peaks"].cpu().detach()
        }

    def configure_optimizers(self):
        optimizer = AdamW(
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
