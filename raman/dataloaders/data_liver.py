from typing import Optional
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed import is_available, is_initialized

import numpy as np
import logging
import lightning as L
import raman.dataloaders.scripts.augmentations as aug
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
from raman.dataloaders.scripts.slicer import window_slicer as slicer
import os
import json

logger = logging.getLogger(__name__)

def use_ddp():
    return is_available() and is_initialized()

def normalization(data):
    minVals = np.min(data, axis=0)
    maxVals = np.max(data, axis=0)
    ranges = maxVals - minVals
    return (data - minVals) / ranges

def z_score(data):
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    return data

def vector_normalization(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return data / norms

def generate_wavenumbers(start=500, end=2000, num_points=890):
    return np.linspace(start, end, num_points, dtype=np.float32)

def global_wv_min_max_scaling(data):
    script_dir = os.path.dirname(__file__)  # Folder of the current script
    file_path = os.path.join(script_dir, "global_wavenumber_range.json")
    print(os.path.exists(file_path))
    if not os.path.exists(file_path):
        global_wavenumber_range()
 
    with open(file_path, "r") as f:
        min_max = json.load(f)
    min_vals = np.float32(np.array(min_max["min"]))
    max_vals = np.float32(np.array(min_max["max"])) 
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data

class LiverDataset(Dataset):
    def __init__(self, data_file: str, preprocess: str = "z_score", augment: bool = True, sliced: bool = False, 
                 window_ratio: float = 1.0, stride: int = 1, dilation: int = 1):
        
        data = np.loadtxt(data_file, delimiter='\t')
        self.x = data[:, 1:].astype(np.float32)
        self.y = data[:, 0].astype(np.int64)
        self.augment = augment
        # Generate wavenumbers
        self.wavenumbers = generate_wavenumbers(num_points=self.x.shape[1])
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)
        self.sliced = sliced
        self.window_ratio = window_ratio
        self.stride = stride
        self.dilation = dilation
        if self.sliced:
            logger.info("Slicing is enabled.")

        if preprocess == "z_score":
            self.x = z_score(self.x)
        elif preprocess == "normalization":
            self.x = normalization(self.x)
        elif preprocess == "vector":
            self.x = vector_normalization(self.x)
        else:
           logger.info("No preprocess was used.")

    def __len__(self):
        if self.sliced:
            first_sample = np.zeros(shape=(2, self.x[0].shape[0]))
            window_size = int(first_sample.shape[-1] * self.window_ratio)
            self.window_num = slicer(first_sample, window_size=window_size,
                                     stride=self.stride, dilation=self.dilation).shape[1]
            return self.x.shape[0] * self.window_num
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        # Stack wavenumbers with spectra
        if self.sliced:
            origin_idx, win_idx = idx // self.window_num, idx % self.window_num
            intensities = self.x[origin_idx]
            label = self.y[origin_idx]
        else:
            intensities = self.x[idx]
            label = self.y[idx]

        if self.augment:
            intensities = aug.random_augment(intensities)
        data = np.vstack((self.wavenumbers, intensities))

        if self.sliced:
            window_size = int(data.shape[-1] * self.window_ratio)
            data = slicer(data, window_size=window_size, stride=self.stride, dilation=self.dilation)
            data = data[:, win_idx, :]

        return {"x": data, "y": label}


class RamanLiverDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
        preprocess: str = "z_score",
        data_dir: str = "datasets/data_liver",
        train_file: str = "demo data_Raman liver_train.txt",
        val_file: str = "demo data_Raman liver_val.txt",
        test_file: str = "demo data_Raman liver_test.txt",
        augment: bool = True,
        sliced: bool = False,
        window_ratio: float = 1.0,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.augment = augment
        self.sliced = sliced
        self.window_ratio = window_ratio
        self.stride = stride
        self.dilation = dilation

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LiverDataset(
            data_file=f"{self.data_dir}/{self.train_file}", preprocess=self.preprocess, augment=self.augment,
            sliced=self.sliced, window_ratio=self.window_ratio, stride=self.stride, dilation=self.dilation
        )
        self.val_dataset = LiverDataset(
            data_file=f"{self.data_dir}/{self.val_file}", preprocess=self.preprocess, augment=self.augment,
            sliced=self.sliced, window_ratio=self.window_ratio, stride=self.stride, dilation=self.dilation
        )
        self.test_dataset = LiverDataset(
            data_file=f"{self.data_dir}/{self.test_file}", preprocess=self.preprocess, augment=self.augment,
            sliced=self.sliced, window_ratio=self.window_ratio, stride=self.stride, dilation=self.dilation
        )

        print(f"Liver train dataset size: {self.train_dataset.__len__()}")
        print(f"Liver validation dataset size: {self.val_dataset.__len__()}")
        print(f"Liver test dataset size: {self.test_dataset.__len__()}")

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
