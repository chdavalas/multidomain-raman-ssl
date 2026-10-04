from torch import randperm, arange
from torch.distributed import is_available, is_initialized
from typing import Optional
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
import logging
import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
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

# --- Dataset class ---
class MineralMixturesDataset(Dataset):
    def __init__(self, excel_path, preprocess="z_score", augment=False, balance=False):
        self.augment = augment
        self.preprocess = preprocess

        # Load all sheets
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        all_spectra, all_labels, pure_spectra, pure_labels = [], [], [], []
        component_names = None

        for sheet in sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet)

            # Component names
            gr_size_idx = df.columns.get_loc("gr_size") if "gr_size" in df.columns else -1
            wavenumber_idx = df.columns.get_loc("wavenumber") if "wavenumber" in df.columns else len(df.columns)

            if gr_size_idx != -1 and wavenumber_idx != len(df.columns):
                comp_cols = df.columns[gr_size_idx + 1 : wavenumber_idx]
                if component_names is None:
                    component_names = list(comp_cols)

            # Spectral columns = integers between 300–3200
            spectral_cols = []
            for col in df.columns:
                try:
                    w = int(col)
                    if 300 <= w <= 3200:
                        spectral_cols.append(col)
                except ValueError:
                    continue
            spectral_cols = sorted(spectral_cols, key=lambda x: int(x))

            for _, row in df.iterrows():
                if pd.isna(row[spectral_cols[0]]):
                    continue

                spectrum = row[spectral_cols].values.astype(np.float32)

                # Build label vector
                label_vec = np.zeros(len(component_names), dtype=int)
                for i, comp in enumerate(component_names):
                    if comp in row and not pd.isna(row[comp]) and row[comp] != 0:
                        label_vec[i] = 1

                all_spectra.append(spectrum)
                all_labels.append(label_vec)

                # Pure check
                is_pure = ("project" in row and row["project"] == "pure") and (np.sum(label_vec) == 1)
                if is_pure:
                    pure_spectra.append(spectrum)
                    pure_labels.append(label_vec)

        self.query_samples = np.array(all_spectra)  # mixtures + pure
        self.labels = np.array(all_labels)          # shape (358, 20)
        self.reference_samples = np.array(pure_spectra)
        self.reference_labels = np.array(pure_labels)
        self.component_names = component_names
        self.wavenumbers = np.array([int(c) for c in spectral_cols], dtype=np.float32)

        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

        # Preprocess
        if preprocess == "z_score":
            self.query_samples = np.array([z_score(s) for s in self.query_samples])
            self.reference_samples = np.array([z_score(s) for s in self.reference_samples])
        elif preprocess == "normalization":
            self.query_samples = np.array([normalization(s) for s in self.query_samples])
            self.reference_samples = np.array([normalization(s) for s in self.reference_samples])
        elif preprocess == "vector":
            self.query_samples = vector_normalization(self.query_samples)
            self.reference_samples = vector_normalization(self.reference_samples)

        # Build pairs
        all_pairs, all_targets = [], []
        for i, query_spec in enumerate(self.query_samples):
            label_vec = self.labels[i]
            for j, ref_spec in enumerate(self.reference_samples):
                comp_idx = np.argmax(self.reference_labels[j])  # which component this ref represents
                y = int(label_vec[comp_idx])  # 1 if present, 0 if absent
                all_pairs.append((query_spec, ref_spec, self.wavenumbers))
                all_targets.append(y)

        # Balance
        if balance:
            pos_pairs = [(p, y) for p, y in zip(all_pairs, all_targets) if y == 1]
            neg_pairs = [(p, y) for p, y in zip(all_pairs, all_targets) if y == 0]

            len_pos, len_neg = len(pos_pairs), len(neg_pairs)
            if len_pos > len_neg:
                extra_neg = random.choices(neg_pairs, k=len_pos - len_neg)
                neg_pairs = neg_pairs + extra_neg
            elif len_neg > len_pos:
                extra_pos = random.choices(pos_pairs, k=len_neg - len_pos)
                pos_pairs = pos_pairs + extra_pos

            balanced = pos_pairs + neg_pairs
            random.shuffle(balanced)

            self.pairs = [p for p, _ in balanced]
            self.targets = [y for _, y in balanced]
        else:
            self.pairs = all_pairs
            self.targets = all_targets

        logger.info(
            f"Loaded {len(self.pairs)} pairs: "
            f"{sum(self.targets)} positive, {len(self.targets) - sum(self.targets)} negative."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query_spec, ref_spec, wv = self.pairs[idx]
        y = self.targets[idx]
        x1 = np.stack([wv, query_spec], axis=0).astype(np.float32)
        x2 = np.stack([wv, ref_spec], axis=0).astype(np.float32)
        return {"x1": x1, "x2": x2, "y": y}


class MineralMixturesDataModule(L.LightningDataModule):
    def __init__(self, excel_path, batch_size=32, num_workers=1, pin_memory=False, preprocess="z_score", augment=False, balance=False, val_split=0.1, test_split=0.1):
        super().__init__()
        self.excel_path = excel_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.augment = augment
        self.balance = balance
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: Optional[str] = None):
        dataset = MineralMixturesDataset(
            excel_path=self.excel_path,
            preprocess=self.preprocess,
            augment=self.augment,
            balance=self.balance,
        )
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print(f"Mineral Mixtures Train dataset size: {len(self.train_dataset)}")
        print(f"Mineral Mixtures Validation dataset size: {len(self.val_dataset)}")
        print(f"Mineral Mixtures Test dataset size: {len(self.test_dataset)}")

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
