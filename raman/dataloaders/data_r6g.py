from torch import randperm, arange
from torch.distributed import is_available, is_initialized

from typing import Optional
from torch.utils.data import Dataset, DataLoader,  random_split, DistributedSampler
from itertools import combinations, product
import raman.dataloaders.scripts.augmentations as aug
import lightning as L
import torch
import pandas as pd
import numpy as np
import logging
import os 
import random
import matplotlib.pyplot as plt
import json
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range

logger = logging.getLogger(__name__)
def use_ddp():
    return is_available() and is_initialized()

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

def file_min_max_normalization(spectra: np.ndarray) -> np.ndarray:
    """
    Normalizes all spectra from a single CSV file together into [0, 1].
    Expects spectra shape = (num_spectra, num_points).
    Returns normalized spectra of same shape.
    """
    min_val = np.min(spectra)
    max_val = np.max(spectra)
    if max_val == min_val:
        return np.zeros_like(spectra)  # avoid div by zero, all values same
    return (spectra - min_val) / (max_val - min_val)

# def wavenumber_normalization(wavenumbers: np.ndarray) -> np.ndarray:
#     """
#     Normalizes wavenumbers to [0, 1] based on fixed range 500-2000 cm^-1 because R6G wavenumbers lie within this range.
#     """
#     return (wavenumbers - 500) / (2000 - 500)

def save_spectrum_plot(wavenumbers, spectrum, out_dir, filename_prefix):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(wavenumbers, spectrum)
    plt.xlabel("Wavenumber (normalized)", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.title(filename_prefix, fontsize=13)
    plt.tight_layout()

    filepath = os.path.join(out_dir, f"{filename_prefix}.png")
    plt.savefig(filepath)
    plt.close()

class R6GDataset(Dataset):
    def __init__(self, root_dir, reference_dir, positive_dir, negative_dir,
                 preprocess="z_score", augment=False, balance=False):
        self.mix_samples = []   # (wavenumber, spectrum)
        self.mix_labels = []    # 1 = positive, 0 = negative
        self.augment = augment
        self.preprocess = preprocess
        self.reference_samples = []

        # ----------------
        # Load mixtures
        # ----------------
        def load_mixtures(subdir, label):
            class_dir = os.path.join(root_dir, subdir)
            for filename in os.listdir(class_dir):
                if not filename.endswith(".csv"):
                    continue
                df = pd.read_csv(os.path.join(class_dir, filename)).dropna()
                wavenumbers = df["Raman shift"].to_numpy(dtype=np.float64)
                drop_cols = ['Raman shift'] + [c for c in df.columns if c.startswith("St.Dev") or c.startswith("average")]
                df = df.drop(columns=drop_cols, errors='ignore')
                spectra = df.to_numpy(dtype=np.float64).T  # [num_spectra, len_wv]
                sorted_idx = np.argsort(wavenumbers)
                wavenumbers = global_wv_min_max_scaling(wavenumbers)

                for spectrum in spectra:
                    spec = spectrum[sorted_idx]
                    wv_sorted = wavenumbers[sorted_idx].astype(np.float32)
                    # spec = np.where(spec<0.0, 0.0, spec)
                    self.mix_samples.append((wv_sorted, spec.astype(np.float32)))
                    self.mix_labels.append(label)
                

        # ----------------
        # Load reference spectra
        # ----------------
        def load_ref_dir(reference_dir):
            ref_dir = os.path.join(root_dir, reference_dir)

            # Only load the first and the third CSV file found in the reference directory 
            # (PLS_CuO_Si_40nmAg_514nm_0.5s_1lp) 
            # (PLS_CuO_Si_140nmAg_514nm_0.5s_1lp)
            # Using [0:3:2] to load only first and third files due to issues with second file   
            
            for filename in os.listdir(ref_dir)[0:3:2]: 
                if not filename.endswith(".csv"):
                    continue
                df = pd.read_csv(os.path.join(ref_dir, filename)).dropna()
                ref_wavenumbers = df["Raman shift"].to_numpy(dtype=np.float64)
                drop_cols = ['Raman shift'] + [c for c in df.columns if c.startswith("St.Dev") or c.startswith("average")]
                df = df.drop(columns=drop_cols, errors='ignore')
                ref_spectra = df.to_numpy(dtype=np.float64).T
                sorted_idx = np.argsort(ref_wavenumbers)
                ref_wavenumbers = global_wv_min_max_scaling(ref_wavenumbers)
                
                wv_sorted = ref_wavenumbers[sorted_idx].astype(np.float32)
                for spectrum in ref_spectra:
                    spec = spectrum[sorted_idx]
                    # spec = np.where(spec<0.0, 0.0, spec)
                    self.reference_samples.append((wv_sorted, spec.astype(np.float32)))


        # Load positive and negative dirs
        load_mixtures(positive_dir, label=1)
        load_mixtures(negative_dir, label=0)

        self.mix_samples = np.array(self.mix_samples)
        self.mix_labels = np.array(self.mix_labels)

        # Load reference spectra
        load_ref_dir(reference_dir)
        self.reference_samples = np.array(self.reference_samples)



        # Apply preprocessing consistently to both reference spectra and mixture spectra
        if preprocess == "file_min_max":
            logger.info("Applying per-file min-max normalization (0–1) to reference and mixture spectra.")
            self.mix_samples[:, 1, :]       = file_min_max_normalization(self.mix_samples[:, 1, :])
            self.reference_samples[:, 1, :] = file_min_max_normalization(self.reference_samples[:, 1, :])

        elif preprocess == "z_score":
            logger.info("Applying Z-score normalization to reference and mixture spectra.")
            self.reference_samples[:, 1, :] = z_score(self.reference_samples[:, 1, :])
            self.mix_samples[:, 1, :]       = z_score(self.mix_samples[:, 1, :])

        elif preprocess == "normalization":
            logger.info("Applying L1/L2 normalization to reference and mixture spectra.")
            self.reference_samples[:, 1, :] = normalization(self.reference_samples[:, 1, :])
            self.mix_samples[:, 1, :]       = normalization(self.mix_samples[:, 1, :])

        elif preprocess == "global_normalization":
            logger.info("Applying L1/L2 normalization to reference and mixture spectra.")
            glob_min = min(np.min(self.reference_samples[:, 1, :]), np.min(self.mix_samples[:, 1, :]))
            glob_max = max(np.max(self.reference_samples[:, 1, :]), np.max(self.mix_samples[:, 1, :]))
            self.reference_samples[:, 1, :] = self.reference_samples[:, 1, :] - glob_min
            self.reference_samples[:, 1, :] = self.reference_samples[:, 1, :] / (glob_max - glob_min)
            self.mix_samples[:, 1, :]       = self.mix_samples[:, 1, :] - glob_min
            self.mix_samples[:, 1, :]       = self.mix_samples[:, 1, :] / (glob_max - glob_min)

        elif preprocess == "vector":
            logger.info("Applying vector normalization to reference and mixture spectra.")
            self.reference_samples[:, 1, :] = vector_normalization(self.reference_samples[:, 1, :])
            self.mix_samples[:, 1, :]       = vector_normalization(self.mix_samples[:, 1, :])

        else:
            logger.info("No preprocessing applied to reference or mixture spectra.")


        # ----------------
        # Generate pairs
        # ----------------
        all_pairs, all_targets = [], []
        for i, (wv_q, spec_q) in enumerate(self.mix_samples):
            label = self.mix_labels[i]
            for wv_r, spec_r in self.reference_samples:
                all_pairs.append(((wv_q, spec_q), (wv_r, spec_r)))
                all_targets.append(label)

        # Balance if requested
        if balance:
            pos_pairs = [(p, l) for p, l in zip(all_pairs, all_targets) if l == 1]
            neg_pairs = [(p, l) for p, l in zip(all_pairs, all_targets) if l == 0]
            len_pos, len_neg = len(pos_pairs), len(neg_pairs)
            if len_pos > len_neg:
                extra_neg = random.choices(neg_pairs, k=len_pos - len_neg)
                neg_pairs += extra_neg
            elif len_neg > len_pos:
                extra_pos = random.choices(pos_pairs, k=len_neg - len_pos)
                pos_pairs += extra_pos
            balanced = pos_pairs + neg_pairs
            random.shuffle(balanced)
            self.pairs = [p for p, _ in balanced]
            self.targets = [l for _, l in balanced]
        else:
            self.pairs = all_pairs
            self.targets = all_targets

        logger.info(f"Loaded {len(self.pairs)} pairs: {sum(self.targets)} positive, {len(self.targets)-sum(self.targets)} negative.")

    def __len__(self):
        print( f"Dataset length: {len(self.pairs)}")
        return len(self.pairs)

    def __getitem__(self, idx):
        (wv1, spec1), (wv2, spec2) = self.pairs[idx]

        mix_idx = idx // len(self.mix_samples)  # Group by mixture

        if self.augment:
            spec1 = aug.random_augment(spec1)
            spec2 = aug.random_augment(spec2)

        x1 = np.stack([wv1, spec1], axis=0)
        x2 = np.stack([wv2, spec2], axis=0)
        y = self.targets[idx]

        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        y  = torch.tensor(y, dtype=torch.long)

        return {"x": x1, "ref": x2, "y": y, "mixture_index": mix_idx}




class R6GDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        reference_dir: str,
        positive_dir: str ,
        negative_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        preprocess: str = "z_score",
        augment: bool = False,
        balance: bool = False,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.reference_dir = reference_dir
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.augment = augment
        self.balance = balance
        self.test_split = test_split
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        dataset = R6GDataset(root_dir=self.data_dir, reference_dir=self.reference_dir, 
                             positive_dir=self.positive_dir, negative_dir=self.negative_dir, 
                             preprocess=self.preprocess,augment=self.augment,balance=self.balance,)
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Validation size: {len(self.val_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")


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
