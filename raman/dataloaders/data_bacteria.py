from typing import Optional
from torch.distributed import is_available, is_initialized
from torch.utils.data import DataLoader, Dataset, random_split
import logging
import numpy as np
import os
import lightning as L
import raman.dataloaders.scripts.augmentations as aug
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
from raman.dataloaders.scripts.slicer import window_slicer as slicer
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler

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

class BacteriaDataset(Dataset):
    def __init__(
        self,
        data_dir="./datasets/bacteria",
        wavenumbers_file="wavenumbers.npy",
        data_file="X_reference.npy",
        labels_file="y_reference.npy",
        preprocess: str = "z_score",
        augment: bool = False,
        sliced: bool = False, 
        window_ratio: float = 0.9,
        stride: int = 3, 
        dilation: int = 1,
        sampling_ratio: float = 1.0
    ):
        # Load wavenumbers
        self.sampling_ratio = sampling_ratio

        self.wavenumbers = np.load(
            os.path.join(data_dir, wavenumbers_file), allow_pickle=True
        )
        self.sorted_indices = np.argsort(
            self.wavenumbers
        )  # Get indices for ascending order
        self.wavenumbers = self.wavenumbers[self.sorted_indices].astype(np.float32)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

        # Load intensity data and labels
        self.data = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
        self.labels = np.load(os.path.join(data_dir, labels_file), allow_pickle=True)

        if sampling_ratio < 1.0:
            num_samples = int(self.data.shape[0] * self.sampling_ratio)
            indices = np.random.choice(self.data.shape[0], num_samples, replace=False)
            self.data = self.data[indices]
            self.labels = self.labels[indices]

        logger.info(f"Loaded {self.data.shape[0]} samples with {self.data.shape[1]} features.")

        self.sliced = sliced
        self.window_ratio = window_ratio
        self.dilation = dilation
        self.stride = stride

        # Preprocess intensity data
        if preprocess == "z_score":
            self.data = z_score(self.data)
        elif preprocess == "normalization":
            self.data = normalization(self.data)
        else:
            raise ValueError(f"Unknown preprocessing method: {preprocess}")

        self.augment = augment
        if self.augment:
            logger.info("Augmentation is enabled.")


    def __len__(self):
        if self.sliced:
            logger.info("Slicing is enabled.")
            first_sample = np.zeros(shape=(2,self.data.shape[-1]))
            window_size = int(first_sample.shape[-1]*self.window_ratio)
            self.window_num = slicer(first_sample, window_size=window_size, 
                                     stride=self.stride, dilation=self.dilation).shape[1]
            
            return int(self.data.shape[0] * self.window_num)
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        if self.sliced:
            origin_idx, win_idx = idx//self.window_num, idx%self.window_num
            intensities = self.data[origin_idx]
            label = self.labels[origin_idx].astype(np.int64)
        else:
            intensities = self.data[idx]
            label = self.labels[idx].astype(np.int64)

        intensities = intensities[self.sorted_indices].astype(np.float32)

        if self.augment:
            intensities = aug.random_augment(intensities)

        # Stack wavenumbers and intensities
        data = np.stack([self.wavenumbers, intensities], axis=0)

        if self.sliced:
            window_size=int(intensities.shape[-1]*self.window_ratio)
            data = slicer(data, window_size=window_size, stride=self.stride, dilation=self.dilation)
            data = data[:, win_idx, :]

        return { "x": data, "y": label }


class BacteriaDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        val_split: float = 0.1,
        data_dir: str = "./datasets/bacteria",
        wavenumbers_file: str = "wavenumbers.npy",
        data_file: str = "X_reference.npy",
        labels_file: str = "y_reference.npy",
        test_data_file: str = "X_test.npy",
        test_labels_file: str = "y_test.npy",
        preprocess: str = "z_score",
        augment: bool = True,
        sliced: bool = False,
        window_ratio: float = 0.9,
        stride: int = 3,
        dilation: int = 1,
        sampling_ratio: float = 1.0
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.data_dir = data_dir
        self.wavenumbers_file = wavenumbers_file
        self.data_file = data_file
        self.labels_file = labels_file
        self.test_data_file = test_data_file
        self.test_labels_file = test_labels_file
        self.preprocess = preprocess
        self.augment = augment
        self.sliced = sliced
        self.window_ratio = window_ratio
        self.stride = stride
        self.dilation = dilation
        self.sampling_ratio = sampling_ratio

    def setup(self, stage: Optional[str] = None):
        dataset = BacteriaDataset(
            data_dir=self.data_dir,
            wavenumbers_file=self.wavenumbers_file,
            data_file=self.data_file,
            labels_file=self.labels_file,
            preprocess=self.preprocess,
            augment=self.augment,
            sliced=self.sliced,
            window_ratio=self.window_ratio,
            stride=self.stride,
            dilation=self.dilation,
            sampling_ratio=self.sampling_ratio
        )

        test_dataset = BacteriaDataset(
            data_dir=self.data_dir,
            wavenumbers_file=self.wavenumbers_file,
            data_file=self.test_data_file,
            labels_file=self.test_labels_file,
            preprocess=self.preprocess,
            augment=self.augment,
            sliced=self.sliced,
            window_ratio=self.window_ratio,
            stride=self.stride,
            dilation=self.dilation,
            sampling_ratio=self.sampling_ratio
        )

        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(test_dataset))
        train_size = len(dataset) - val_size


        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        self.test_dataset = test_dataset
        
        print(f"Bacteria train dataset size: {train_size}")
        print(f"Bacteria validation dataset size: {val_size}")
        print(f"Bacteria test dataset size: {test_size}")

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
