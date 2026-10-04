from torch import randperm, arange
from torch.distributed import is_available, is_initialized
from typing import Optional
from torch.utils.data import DataLoader, Dataset, random_split
import logging
import os
import numpy as np
import lightning as L
import raman.dataloaders.scripts.augmentations as aug
import json
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
    print(os.path.exists(file_path))
    if not os.path.exists(file_path):
        global_wavenumber_range()
 
    with open(file_path, "r") as f:
        min_max = json.load(f)
    min_vals = np.float32(np.array(min_max["min"]))
    max_vals = np.float32(np.array(min_max["max"])) 
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data

class SangXianchengDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            preprocess="z_score", 
            augment: bool = True, 
            sliced: bool = False,
            window_ratio: float = 1.0,
            stride: int = 3,
            dilation: int = 1):

        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.wavenumbers = []
        self.intensities = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.preprocess = preprocess
        self.transform = True
        self.augment = augment
        self.sliced = sliced
        self.ratio = window_ratio
        self.stride = stride
        self.dilation = dilation

        if "Actinolite" not in self.classes:
            logger.warning("Required folder 'Actinolite' not found in %s.", data_dir)
            raise FileNotFoundError(f"Required class 'Actinolite' not found in {data_dir}")

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.txt'):
                        self.wavenumbers.append(np.loadtxt(
                            os.path.join(cls_dir, img_name), dtype=float, comments='#', delimiter=',', unpack=True)[0])
                        self.intensities.append(np.loadtxt(
                            os.path.join(cls_dir, img_name), dtype=float, comments='#', delimiter=',', unpack=True)[1])
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.class_to_idx[cls_name])

        logger.info("Loading dataset from %s", data_dir)
        logger.info("Number of classes: %d", len(self.classes))
        logger.info("Number of samples: %d", len(self.image_paths))
        
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

        if self.preprocess == "z_score":
            self.intensities = z_score(self.intensities)
        elif self.preprocess == "normalization":
            self.intensities = normalization(self.intensities)
        else:
            logger.info(f"No preprocessing method in [z_score, normalization] is selected. Using raw data.")

        if self.augment:    
            logger.info("Augmentation is enabled.")

    def __len__(self):
        if self.sliced:
            logger.info("Slicing is enabled.")
            first_sample = np.zeros(shape=(2,self.wavenumbers[0].shape[0]))
            window_size = int(first_sample.shape[-1]*self.ratio)
            self.window_num = slicer(first_sample, window_size=window_size, 
                                     stride=self.stride, dilation=self.dilation).shape[1]
            return len(self.image_paths)*self.window_num
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.sliced:
            origin_idx, win_idx = idx//self.window_num, idx%self.window_num
            wavenumbers = self.wavenumbers[origin_idx]
            intensities = self.intensities[origin_idx]
            label = self.labels[origin_idx]
        else:
            wavenumbers = self.wavenumbers[idx]
            intensities = self.intensities[idx]
            label = self.labels[idx]

        if self.augment:
            intensities = aug.random_augment(intensities)

        data = np.stack((wavenumbers, intensities), axis=0) 
        data = data.astype(np.float32) 

        if self.sliced:
            window_size=int(data.shape[-1]*self.ratio)
            data = slicer(data, window_size=window_size, stride=self.stride, dilation=self.dilation)
            data = data[:, win_idx, :]

        return { "x": data, "y": label }

class SangXianchengDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "datasets/SangXiancheng",
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        val_split: float = 0.1,
        test_split: float = 0.1,
        preprocess: str = "none",
        augment: bool = True,
        sliced: bool = False,
        window_ratio: float = 1.0,
        stride: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.preprocess = preprocess
        self.augment = augment
        self.sliced = sliced
        self.ratio = window_ratio
        self.stride = stride
        self.dilation = dilation

    def setup(self, stage: Optional[str] = None):
        dataset = SangXianchengDataset(self.data_dir, self.preprocess, augment=self.augment, sliced=self.sliced, 
                                       window_ratio=self.ratio, stride=self.stride, dilation=self.dilation)
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print(f"SangXiancheng Train dataset size: {train_size}")
        print(f"SangXiancheng Validation dataset size: {val_size}")
        print(f"SangXiancheng Test dataset size: {test_size}")

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
