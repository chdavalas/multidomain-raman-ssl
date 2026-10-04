from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DenoisingDataset(Dataset):
    """
    Wraps a base dataset. 
    Returns:
        x: Noisy Spectrum [2, T] (Wavenumbers, Noisy Intensity)
        y: Clean Spectrum [2, T] (Wavenumbers, Clean Intensity)
    """
    def __init__(self, base_dataset: Dataset, noise_std: float = 0.1, noise_type: str = "gaussian"):
        self.base_dataset = base_dataset
        self.noise_std = noise_std
        self.noise_type = noise_type

    def __len__(self):
        return len(self.base_dataset)

    def add_noise(self, intensities):
        """Adds synthetic noise to intensities."""
        if self.noise_type == "gaussian":
            noise = torch.randn_like(intensities) * self.noise_std
            return intensities + noise
        return intensities

    def __getitem__(self, idx):
        # Get the "Clean" sample from the base dataset
        base_sample = self.base_dataset[idx]
        clean_signal = base_sample["x"]
        
        if isinstance(clean_signal, np.ndarray):
            clean_signal = torch.from_numpy(clean_signal)
            
        # Ensure it is float32
        clean_signal = clean_signal.float()

        # Separate Wavenumbers and Intensities
        wavenumbers = clean_signal[0, :]
        clean_intensities = clean_signal[1, :]

        # Generate Noisy Input
        noisy_intensities = self.add_noise(clean_intensities)
        
        # Stack back to [2, T]
        # x = [Wavenumbers, Noisy]
        x = torch.stack([wavenumbers, noisy_intensities], dim=0)
        
        # y = [Wavenumbers, Clean]
        y = torch.stack([wavenumbers, clean_intensities], dim=0)

        return {"x": x, "y": y}


class DenoisingDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_datamodule: L.LightningDataModule,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = False,
        noise_std: float = 0.05,
        noise_type: str = "gaussian"
    ):
        super().__init__()        
        self.base_datamodule = base_datamodule
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.noise_std = noise_std
        self.noise_type = noise_type

    def setup(self, stage: Optional[str] = None):
        # 1. Setup the base data module first (loads files, splits, etc.)
        self.base_datamodule.setup(stage)

        self.train_dataset = DenoisingDataset(self.base_datamodule.train_dataset, noise_std=self.noise_std, noise_type=self.noise_type)
        self.val_dataset = DenoisingDataset(self.base_datamodule.val_dataset, noise_std=self.noise_std,noise_type=self.noise_type)
        self.test_dataset = DenoisingDataset(self.base_datamodule.test_dataset, noise_std=self.noise_std,noise_type=self.noise_type)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset if hasattr(self, 'predict_dataset') else self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )