from torch import randperm, arange
from torch.distributed import is_available, is_initialized
from typing import Optional
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
import logging
import numpy as np
import pandas as pd
import numpy as np
import lightning as L
import raman.dataloaders.scripts.augmentations as aug
import json
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
import os
from raman.dataloaders.scripts.slicer import window_slicer as slicer
import random
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

class AdenineDataset(Dataset):
    def __init__(self, data_file="./datasets/Adenine/ILSdata.csv", 
                 concentration_level=0.8, test_dataset=False, 
                 preprocess: str = "z_score", augment: bool = True,
                    sliced: bool = False, window_ratio: float = 0.9,
                    stride: int = 3, dilation: int = 1
                 
    ):

        df = pd.read_csv(data_file, header=None)
        self.wavenumbers = df.iloc[0, 9:].to_numpy(dtype=np.float32)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

        self.sliced = sliced
        self.ratio = window_ratio
        self.dilation = dilation
        self.stride = stride

        df = df.iloc[1:].reset_index(drop=True)

        if test_dataset:
            df = df[df[5] == "test"]
        else:
            df = df[df[5].isin(["train", "blank"])]

        df = df.replace([np.nan, np.inf, -np.inf], 0.0)

        self.labels = (df[6].astype(float).to_numpy() >= concentration_level).astype(int)
        self.intensities = df.iloc[:, 9:].to_numpy(dtype=np.float32)
        
        if preprocess == "z_score":
            self.intensities = z_score(self.intensities)
        elif preprocess == "normalization":
            self.intensities = normalization(self.intensities)
        elif preprocess == "vector":
            self.x = vector_normalization(self.x)
        else:
            logger.info("No preprocess was used.")
        
        self.augment = augment
        if self.augment:
            logger.info("Augmentation is enabled.")

    def __len__(self):
        if self.sliced:
            logger.info("Slicing is enabled.")
            first_sample = np.zeros(shape=(2,self.intensities[0].shape[0]))
            window_size = int(first_sample.shape[-1]*self.ratio)
            self.window_num = slicer(first_sample, window_size=window_size, 
                                     stride=self.stride, dilation=self.dilation).shape[1]
            return self.intensities.shape[0]*self.window_num
        return self.intensities.shape[0]

    def __getitem__(self, idx):

        if self.sliced:
            origin_idx, win_idx = idx//self.window_num, idx%self.window_num
            intensities = self.intensities[origin_idx]
            label = self.labels[origin_idx]

        else:
            intensities = self.intensities[idx]
            label = self.labels[idx]


        if self.augment:
            intensities = aug.random_augment(intensities)

        data  = np.stack((self.wavenumbers, intensities))

        if self.sliced:
            window_size=int(data.shape[-1]*self.ratio)
            data = slicer(data, window_size=window_size, stride=self.stride, dilation=self.dilation)
            data = data[:, win_idx, :]

        return {"x": data, "y": label}


class AdenineDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        val_split: float = 0.1,
        data_file: str = "./datasets/Adenine/ILSdata.csv",
        concentration_level: float = 0.8,
        preprocess: str = "z_score",
        augment: bool = True,
        sliced: bool = False,
        window_ratio: float = 0.9,
        stride: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.val_split = val_split
        self.data_file = data_file
        self.concentration_level = concentration_level
        self.preprocess = preprocess
        self.augment = augment
        self.sliced = sliced
        self.window_ratio = window_ratio
        self.stride = stride
        self.dilation = dilation

    def setup(self, stage: Optional[str] = None):
        # build test dataset as in the original paper
        self.test_dataset = AdenineDataset(
            data_file=self.data_file, test_dataset=True, concentration_level=self.concentration_level, 
            preprocess=self.preprocess, augment=self.augment, sliced=self.sliced,
            window_ratio=self.window_ratio, stride=self.stride, dilation=self.dilation
        )

        # build train and separate a few for validation
        dataset = AdenineDataset(data_file=self.data_file, test_dataset=False, 
                                 concentration_level=self.concentration_level, 
                                 preprocess=self.preprocess, augment=self.augment,
                                 sliced=self.sliced, window_ratio=self.window_ratio,
                                 stride=self.stride, dilation=self.dilation)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )


        print(f"Adenine Train dataset size: {train_size}")
        print(f"Adenine Validation dataset size: {val_size}")
        print(f"Adenine Test dataset size: {len(self.test_dataset)}")

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
