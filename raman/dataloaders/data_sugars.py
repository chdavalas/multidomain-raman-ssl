import os
import random
import logging
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
from torch.distributed import is_available, is_initialized

import torch
import pandas as pd
import numpy as np
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import numpy as np
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import json
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range


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

def airPLS_sparse(x, lam=1e5, porder=1, itermax=15, use_savgol=True):
    """
    Adaptive iteratively reweighted penalized least squares (airPLS)
    using sparse Whittaker smoother.

    Returns:
        corrected spectrum (baseline removed)
    """

    x = np.asarray(x, dtype=float)

    # Optional Savitzky–Golay pre-smoothing
    if use_savgol:
        x_smooth = savgol_filter(x, window_length=9, polyorder=3, mode='nearest')
    else:
        x_smooth = x

    m = x_smooth.size
    w = np.ones(m)

    for i in range(1, itermax + 1):
        # Estimate baseline using Whittaker
        baseline = whittaker_smooth_sparse(x_smooth, w, lam, diff_order=porder)

        diff = x_smooth - baseline

        # stopping condition
        dssn = np.abs(diff[diff < 0].sum())
        if dssn < 0.001 * np.abs(x_smooth).sum() or i == itermax:
            break

        # Update weights
        w[diff >= 0] = 0   # Peaks → 0 weight (ignored)
        w[diff < 0]  = np.exp(i * np.abs(diff[diff < 0]) / dssn)

        # Force endpoints to match (stabilizes Raman edges)
        if diff[diff < 0].size > 0:
            w0 = np.exp(i * np.max(diff[diff < 0]) / dssn)
            w[0] = w[-1] = w0

    return x - baseline

def whittaker_smooth_sparse(x, w, lam, diff_order=1):
    """
    Sparse Whittaker-Eilers smoother.
    w: weights (1D array)
    lam: smoothing penalty parameter
    diff_order: usually 1 or 2
    """
    X = np.asarray(x, dtype=float)
    m = X.size

    # Sparse identity
    E = eye(m, format='csc')

    # Difference operator
    D = E[1:] - E[:-1]
    for _ in range(diff_order - 1):
        D = D[1:] - D[:-1]

    W = diags(w, 0, shape=(m, m))
    A = W + lam * (D.T @ D)
    B = W @ X

    background = spsolve(A, B)
    return np.asarray(background)

def get_concentration_level(value):
    """Convert concentration value to level according to specified ranges."""
    # if 0 <= value <= 30:
    #     return 0  # Level 1 (0-indexed for PyTorch)
    # elif 30 < value <= 75:
    #     return 1  # Level 2
    # elif 75 < value <= 120:
    #     return 2  # Level 3
    # elif 120 < value <= 375:
    #     return 3  # Level 4
    # # elif 375 < value:
    # #     return 4  # Level 5
    # else:
    #     # Handle values outside specified ranges
    #     if value < 0:
    #         return 0
    #     else:
    #         return 3
        
    if 0 <= value < 30:
        return 0  # Level 1 (0-indexed for PyTorch)
    elif 30 <= value < 75:
        return 1  # Level 2
    elif 75 <= value < 120:
        return 2  # Level 3
    elif 120 <= value < 375:
        return 3  # Level 4
    elif 375 <= value:
        return 4  # Level 5
    else:
        # Handle values outside specified ranges
        if value < 0:
            return 0
        else:
            return 4    

class SugarSiameseDataset(Dataset):
    def __init__(self, spectra_csv, metadata_csv, preprocess="z_score", augment=False, balance=False, task_type="multi_class"):
        self.augment = augment
        self.preprocess = preprocess
        self.task_type = task_type
        
        # Dynamically set num_classes so the balancing loop works correctly
        self.num_classes = 2 if task_type == "binary" else 5

        # --- Load mixture spectra ---
        df = pd.read_csv(spectra_csv)
        self.wavenumbers = df["Wavenumber [cm-1]"].to_numpy(dtype=np.float32)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)  # Scale wavenumbers to [0, 1]

        df = df.drop("Wavenumber [cm-1]", axis=1)
        spectra = df.to_numpy(dtype=np.float32).T  # [num_spectra, len_wavenumbers]

        # --- Select reference spectra (pure components) ---
        ref_indices = [59, 7570, 23, 83]
        self.reference_samples = spectra[ref_indices]

        # --- Mixture spectra (all spectra) ---
        self.query_samples = spectra

        # --- Metadata / Labels ---
        meta_df = pd.read_csv(metadata_csv)
        sugar_components = ["Sucrose", "Fructose", "Maltose", "Glucose"]
        sugar_columns = [f"{sugar} [ul]" for sugar in sugar_components]

        # Store concentration values
        self.concentrations = np.zeros((len(meta_df), len(sugar_columns)))
        for i, sugar_col in enumerate(sugar_columns):
            if sugar_col in meta_df.columns:
                self.concentrations[:, i] = meta_df[sugar_col].to_numpy(dtype=np.float32)
            else:
                logger.warning(f"Column {sugar_col} not found in metadata")
                self.concentrations[:, i] = 0.0

        # Convert concentrations to labels based on the chosen task_type
        self.concentration_levels = np.zeros((len(meta_df), len(sugar_columns)), dtype=np.int64)
        for i in range(len(meta_df)):
            for j in range(len(sugar_columns)):
                val = self.concentrations[i, j]
                
                if self.task_type == "binary":
                    # Label 1 for presence (value > 0), Label 0 for absence (value <= 0)
                    self.concentration_levels[i, j] = 1 if val > 0 else 0
                else:
                    # Original logic for concentration levels 0-4
                    self.concentration_levels[i, j] = get_concentration_level(val)

        # ---------------------------------------------------------
        # Baseline correction (airPLS) + Whittaker smoothing
        # BEFORE any normalization / z-score
        # ---------------------------------------------------------
        processed_query = []
        for s in self.query_samples:
            s_corr = airPLS_sparse(s, lam=1e5, porder=1, itermax=15, use_savgol=True)
            s_smooth = whittaker_smooth_sparse(s_corr, w=np.ones_like(s_corr), lam=1e3, diff_order=2)
            processed_query.append(s_smooth)
        self.query_samples = np.array(processed_query)

        processed_ref = []
        for s in self.reference_samples:
            s_corr = airPLS_sparse(s, lam=1e5, porder=1, itermax=15, use_savgol=True)
            s_smooth = whittaker_smooth_sparse(s_corr, w=np.ones_like(s_corr), lam=1e3, diff_order=2)
            processed_ref.append(s_smooth)
        self.reference_samples = np.array(processed_ref)

        # ---------------------------------------------------------
        # Apply user-chosen preprocessing
        # ---------------------------------------------------------
        if preprocess == "z_score":
            logger.info("Applying z_score normalization to reference and mixture spectra.")
            self.query_samples = np.array([z_score(s) for s in self.query_samples])
            self.reference_samples = np.array([z_score(s) for s in self.reference_samples])
        elif preprocess == "normalization":
            logger.info("Applying standard normalization to reference and mixture spectra.")
            self.query_samples = np.array([normalization(s) for s in self.query_samples])
            self.reference_samples = np.array([normalization(s) for s in self.reference_samples])
        elif preprocess == "vector":
            logger.info("Applying vector normalization to reference and mixture spectra.")
            self.query_samples = vector_normalization(self.query_samples)
            self.reference_samples = vector_normalization(self.reference_samples)
        elif preprocess == "global_normalization":
            logger.info("Applying L1/L2 normalization to reference and mixture spectra.")
            glob_min = min(np.min(self.reference_samples[:, 1, :]), np.min(self.query_samples[:, 1, :]))
            glob_max = max(np.max(self.reference_samples[:, 1, :]), np.max(self.query_samples[:, 1, :]))
            self.reference_samples[:, 1, :] = self.reference_samples[:, 1, :] - glob_min
            self.reference_samples[:, 1, :] = self.reference_samples[:, 1, :] / (glob_max - glob_min)
            self.query_samples[:, 1, :]       = self.query_samples[:, 1, :] - glob_min
            self.query_samples[:, 1, :]       = self.query_samples[:, 1, :] / (glob_max - glob_min)
       
        # --- Build pairs ---
        all_pairs = []
        all_targets = []
        all_concentrations = [] 

        for i, query_spec in enumerate(self.query_samples):
            mixture_levels = self.concentration_levels[i] 
            mixture_values = self.concentrations[i] 
            
            for j, ref_spec in enumerate(self.reference_samples):
                y = mixture_levels[j] 
                actual_concentration = mixture_values[j] 
                
                all_pairs.append((query_spec, ref_spec, self.wavenumbers))
                all_targets.append(y)
                all_concentrations.append(actual_concentration)

        # --- Balance ---
        if balance:
            from collections import Counter
            class_counts = Counter(all_targets)
            max_count = max(class_counts.values())
            
            balanced_pairs = []
            balanced_targets = []
            
            # Oversample minority classes. Uses self.num_classes (2 or 5)
            for class_idx in range(self.num_classes):
                class_indices = [i for i, t in enumerate(all_targets) if t == class_idx]
                if not class_indices:
                    continue
                    
                if len(class_indices) < max_count:
                    for idx in class_indices:
                        balanced_pairs.append(all_pairs[idx])
                        balanced_targets.append(all_targets[idx])
                    
                    oversample_count = max_count - len(class_indices)
                    oversampled_indices = random.choices(class_indices, k=oversample_count)
                    for idx in oversampled_indices:
                        balanced_pairs.append(all_pairs[idx])
                        balanced_targets.append(all_targets[idx])
                else:
                    for idx in class_indices:
                        balanced_pairs.append(all_pairs[idx])
                        balanced_targets.append(all_targets[idx])
            
            combined = list(zip(balanced_pairs, balanced_targets))
            random.shuffle(combined)
            self.pairs = [p for p, _ in combined]
            self.targets = [t for _, t in combined]
        else:
            self.pairs = all_pairs
            self.targets = all_targets

        logger.info(f"Loaded {len(self.pairs)} pairs using {task_type} mode.")
        from collections import Counter
        class_dist = Counter(self.targets)
        for level, count in sorted(class_dist.items()):
            label_name = f"Class {level}" if task_type == "binary" else f"Level {level+1}"
            logger.info(f"  {label_name}: {count} samples")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query_spec, ref_spec, wv = self.pairs[idx]
        y = self.targets[idx]

        x1 = np.stack([wv, query_spec], axis=0).astype(np.float32)
        x2 = np.stack([wv, ref_spec], axis=0).astype(np.float32)

        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        y  = torch.tensor(y, dtype=torch.long)
        return {"x": x1, "ref": x2, "y": y}

class SugarSiameseDataModule(L.LightningDataModule):
    def __init__(self, spectra_csv, metadata_csv, batch_size=32, num_workers=1, pin_memory=False, preprocess="z_score", augment=False, balance=False, val_split=0.1, test_split=0.1, task_type="multi_class"):
        super().__init__()
        self.spectra_csv = spectra_csv
        self.metadata_csv = metadata_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.augment = augment
        self.balance = balance
        self.val_split = val_split
        self.test_split = test_split
        self.task_type = task_type  # <--- New parameter stored

    def setup(self, stage: Optional[str] = None):
        dataset = SugarSiameseDataset(
            spectra_csv=self.spectra_csv,
            metadata_csv=self.metadata_csv,
            preprocess=self.preprocess,
            augment=self.augment,
            balance=self.balance,
            task_type=self.task_type  # <--- Pass to dataset
        )
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print(f"Sugar Siamese Train dataset size: {train_size}")
        print(f"Sugar Siamese Validation dataset size: {val_size}")
        print(f"Sugar Siamese Test dataset size: {test_size}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    
    
class SugarMultiLabelDataset(Dataset):
    def __init__(
        self,
        spectra_csv,
        metadata_csv,
        preprocess="z_score",
        augment=False,
    ):
        self.augment = augment
        self.preprocess = preprocess

        # -----------------------------
        # Load spectra
        # -----------------------------
        df = pd.read_csv(spectra_csv)
        self.wavenumbers = df["Wavenumber [cm-1]"].to_numpy(dtype=np.float32)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)  # Scale wavenumbers to [0, 1]
        
        df = df.drop("Wavenumber [cm-1]", axis=1)

        spectra = df.to_numpy(dtype=np.float32).T  # [num_samples, num_points]
        self.samples = spectra

        # -----------------------------
        # Load metadata / labels
        # -----------------------------
        meta_df = pd.read_csv(metadata_csv)

        self.sugar_components = ["Sucrose", "Fructose", "Maltose", "Glucose"]
        sugar_columns = [f"{s} [ul]" for s in self.sugar_components]

        # concentration values
        concentrations = np.zeros((len(meta_df), len(sugar_columns)), dtype=np.float32)

        for i, col in enumerate(sugar_columns):
            if col in meta_df.columns:
                concentrations[:, i] = meta_df[col].to_numpy(dtype=np.float32)
            else:
                logger.warning(f"Missing column {col}, filling zeros")

        # convert to discrete levels
        labels = np.zeros_like(concentrations, dtype=np.int64)
        for i in range(concentrations.shape[0]):
            for j in range(concentrations.shape[1]):
                labels[i, j] = get_concentration_level(concentrations[i, j])

        self.labels = labels  # shape [N, 5]

        # -----------------------------
        # Baseline correction + smoothing
        # -----------------------------
        processed = []
        for s in self.samples:
            s_corr = airPLS_sparse(s, lam=1e5, porder=1, itermax=15, use_savgol=True)
            s_smooth = whittaker_smooth_sparse(
                s_corr, w=np.ones_like(s_corr), lam=1e3, diff_order=2
            )
            processed.append(s_smooth)

        self.samples = np.array(processed)

        # -----------------------------
        # Preprocessing
        # -----------------------------
        if preprocess == "z_score":
            logger.info("Applying z_score normalization to reference and mixture spectra.")
            self.samples = np.array([z_score(s) for s in self.samples])
        elif preprocess == "normalization":
            logger.info("Applying standard normalization to reference and mixture spectra.")
            self.samples = np.array([normalization(s) for s in self.samples])
        elif preprocess == "vector":
            ogger.info("Applying vector normalization to reference and mixture spectra.")
            self.samples = vector_normalization(self.samples)
        elif preprocess == "global_normalization":
            logger.info("Applying L1/L2 normalization to reference and mixture spectra.")
            glob_min = np.min(self.samples[:, :])
            glob_max = np.max(self.samples[:, :])
            self.samples[:, :] = self.samples[:, :] - glob_min
            self.samples[:, :] = self.samples[:, :] / (glob_max - glob_min)
            
        logger.info(f"Loaded {len(self.samples)} multi-label samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spectrum = self.samples[idx]
        labels = self.labels[idx]

        x = np.stack([self.wavenumbers, spectrum], axis=0).astype(np.float32)
        # x = spectrum.astype(np.float32)[np.newaxis, :]  # or .reshape(1, -1)
        # x = torch.tensor(x, dtype=torch.float32)
        # y  = torch.tensor(y, dtype=torch.long)
        return {
            "x": x,                    # [2, N]
            "y": labels.astype(np.int64)  # [5]
        }


class SugarMultiLabelDataModule(L.LightningDataModule):
    def __init__(
        self,
        spectra_csv,
        metadata_csv,
        batch_size=32,
        num_workers=1,
        pin_memory=False,
        preprocess="z_score",
        augment=False,
        val_split=0.1,
        test_split=0.1,
    ):
        super().__init__()
        self.spectra_csv = spectra_csv
        self.metadata_csv = metadata_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.augment = augment
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: Optional[str] = None):
        dataset = SugarMultiLabelDataset(
            spectra_csv=self.spectra_csv,
            metadata_csv=self.metadata_csv,
            preprocess=self.preprocess,
            augment=self.augment,
        )

        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

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
