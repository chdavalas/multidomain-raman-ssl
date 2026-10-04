from typing import Optional
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributed import is_available, is_initialized

import logging
import numpy as np
import lightning as L
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler, Subset
import os
import json
import raman.dataloaders.scripts.augmentations as aug


logger = logging.getLogger(__name__)

# This dataset is similar to Hyperspectral2 but uses a different data file
# Normalization is already done in the data file

def use_ddp():
    return is_available() and is_initialized()
def global_wv_min_max_scaling(data):
    script_dir = os.path.dirname(__file__)  # Folder of the current script
    file_path = os.path.join(script_dir, "global_wavenumber_range.json")
    if not os.path.exists(file_path):
        global_wavenumber_range()
 
    with open(file_path, "r") as f:
        min_max = json.load(f)
    min_vals = np.float32(np.array(min_max["min"]))
    max_vals = np.float32(np.array(min_max["max"])) 
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data

class Hyperspectral1Dataset(Dataset):
    def __init__(self, augment, data_file='../datasets/hyperspectral/Cell-HR_Norm_500-000-090.npy', wavenumbers_file='datasets/hyperspectral/axis.txt'):
        self.datas = np.load(data_file, allow_pickle=True)
        self.wavenumbers = np.loadtxt(wavenumbers_file, dtype=float, comments='#', unpack=True).astype(np.float32)
        self.augment = augment
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        intensities = self.datas[idx].astype(np.float32)

        if self.augment:
            intensities = aug.random_augment(intensities)

        # Stack wavenumbers and intensities
        data = np.vstack((self.wavenumbers, intensities))  # Shape: (2, sequence_length)
        return {"x": data}

class Hyperspectral1DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        val_split: float = 0.1,
        test_split: float = 0.1,
        data_file: str = '../datasets/hyperspectral/Cell-HR_Norm_500-000-090.npy',
        wavenumbers_file: str = '../datasets/hyperspectral/axis.txt',
        augment: bool = True,
        sampling_ratio: float = 1.0
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.data_file = data_file
        self.wavenumbers_file=wavenumbers_file
        self.augment = augment
        self.sampling_ratio = sampling_ratio

    def setup(self, stage: Optional[str] = None):
        dataset = Hyperspectral1Dataset(augment=self.augment, data_file=self.data_file, wavenumbers_file=self.wavenumbers_file)
        if self.sampling_ratio < 1.0:
            num_samples = int(len(dataset) * self.sampling_ratio)
            dataset = Subset(dataset, np.random.choice(len(dataset), num_samples, replace=False))
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size



        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print(f"Hyperspectral1 Dataset: {len(self.train_dataset)} train samples")
        print(f"Hyperspectral1 Dataset: {len(self.val_dataset)} validation samples")
        print(f"Hyperspectral1 Dataset: {len(self.test_dataset)} test samples")

    def train_dataloader(self):

        if use_ddp():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self):

        if use_ddp():
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):

        if use_ddp():
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.val_dataloader()
