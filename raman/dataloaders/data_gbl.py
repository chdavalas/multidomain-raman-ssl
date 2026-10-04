import os
import random
from torch.distributed import is_available, is_initialized

import logging
import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
import lightning as L
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import raman.dataloaders.scripts.augmentations as aug
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
from raman.dataloaders.scripts.slicer import window_slicer as slicer
import json
import re
from scipy.interpolate import interp1d

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
    script_dir = os.path.dirname(__file__)  
    file_path = os.path.join(script_dir, "global_wavenumber_range.json")
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

        dssn = np.abs(diff[diff < 0].sum())
        if dssn < 0.001 * np.abs(x_smooth).sum() or i == itermax:
            break

        w[diff >= 0] = 0   
        w[diff < 0]  = np.exp(i * np.abs(diff[diff < 0]) / dssn)

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

    E = eye(m, format='csc')
    D = E[1:] - E[:-1]
    for _ in range(diff_order - 1):
        D = D[1:] - D[:-1]

    W = diags(w, 0, shape=(m, m))
    A = W + lam * (D.T @ D)
    B = W @ X

    background = spsolve(A, B)
    return np.asarray(background)

# ==========================================
# Unified Label Extraction Logic
# ==========================================

def extract_concentration_from_dirname(filepath: str) -> float:
    """
    Original GBL logic: Strictly uses the directory name to extract concentration.
    """
    dirname_lower = os.path.basename(os.path.dirname(filepath)).lower()
    
    if 'blank' in dirname_lower:
        return 0.0
    
    level1_patterns = [r'01mgml', r'0\.1mgml', r'0\.1mg', r'0\.1 mg']
    for pattern in level1_patterns:
        if re.search(pattern, dirname_lower):
            return 0.1
    
    if '1mg' in dirname_lower and '01mg' not in dirname_lower and not re.search(r'0\.?1mg', dirname_lower):
        return 1.0
    
    if '10mg' in dirname_lower:
        return 10.0
    
    logger.warning(f"No concentration pattern found in '{dirname_lower}', defaulting to 0.0")
    return 0.0

def get_concentration_level(value: float, custom_bins: Optional[List[float]] = None) -> int:
    """
    Convert a continuous GBL concentration value to a discrete class level.
    Dynamically adapts to any number of thresholds provided in custom_bins.
    """
    bins = custom_bins if custom_bins is not None else [0.05, 0.5, 3.75, 15.0]
    
    # 1. Check the lowest bound (Level 0)
    if value <= bins[0]: 
        return 0          
        
    # 2. Loop through all intermediate bounds (Level 1 to N-1)
    for i in range(len(bins) - 1):
        if bins[i] < value <= bins[i+1]:
            return i + 1
            
    # 3. Check the highest bound (Level N)
    # If the value is greater than the very last bin, it goes to the max class
    return len(bins)

def find_csv_files(root_dir: str, i: int, custom_bins: Optional[List[float]] = None) -> List[Tuple[str, int]]:
    csv_files_with_labels = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                if 'summary' in file.lower():
                    continue
                csv_path = os.path.join(root, file)
                
                # STRICTLY use directory extraction
                raw_concentration = extract_concentration_from_dirname(csv_path)
                concentration_level = get_concentration_level(raw_concentration, custom_bins)
                
                csv_files_with_labels.append((csv_path, concentration_level))
                i += 1
    return csv_files_with_labels

class GBLConcentrationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        preprocess: str = "z_score",
        wavenumber_column: str = "Raman shift",
        augment: bool = True, sliced: bool = False, 
        window_ratio: float = 1.0, stride: int = 1, dilation: int = 1,
        use_reference: bool = False, ref_dir: Optional[str] = None,
        concentration_bins: Optional[List[float]] = None  
    ):
        self.augment = augment
        self.sliced = sliced
        self.window_ratio = window_ratio
        self.stride = stride
        self.dilation = dilation
        self.preprocess = preprocess
        self.wavenumber_column = wavenumber_column
        self.use_reference = use_reference
        self.i = 0
        self.concentration_bins = concentration_bins
        
        if self.sliced:
            logger.info("Slicing is enabled.")
        
        # Pass the bins into the file finder
        csv_files_with_labels = find_csv_files(data_dir, self.i, self.concentration_bins)
        
        if not csv_files_with_labels:
            raise ValueError(f"No CSV files found in directory: {data_dir}")
        
        logger.info(f"Found {len(csv_files_with_labels)} CSV files")
        
        # Load all spectra and labels
        all_spectra = []
        all_labels = []
        self.wavenumbers = None
        
        for csv_idx, (csv_path, concentration_level) in enumerate(csv_files_with_labels):
            try:
                # Load CSV file
                df = pd.read_csv(csv_path)
                
                # Extract wavenumbers (should be the same for all files)
                if self.wavenumbers is None and wavenumber_column in df.columns:
                    self.wavenumbers = df[wavenumber_column].to_numpy(dtype=np.float32)
                    
                drop_cols = ['Raman shift'] + [c for c in df.columns if c.startswith("St.Dev") or c.startswith("average")]
                df = df.drop(columns=drop_cols, errors='ignore')

                intensity_columns = [col for col in df.columns if col != wavenumber_column]
                
                for col in intensity_columns:
                    spectrum = df[col].to_numpy(dtype=np.float32)
                    all_spectra.append(spectrum)
                    all_labels.append(concentration_level)
                    
            except Exception as e:
                logger.warning(f"Error loading {csv_path}: {e}")
                continue
        
        if not all_spectra:
            raise ValueError("No valid spectra loaded from CSV files")
        
        self.samples = np.array(all_spectra, dtype=np.float32)  # [num_samples, length]
        self.labels = np.array(all_labels, dtype=np.int64)      # [num_samples] 
        
        # If wavenumbers not found in any file, create dummy wavenumbers
        if self.wavenumbers is None:
            logger.warning("Wavenumber column not found, creating dummy wavenumbers")
            self.wavenumbers = np.arange(self.samples.shape[1], dtype=np.float32)
        
        # -----------------------------
        # Load Reference Spectra (if flagged)
        # -----------------------------
        if self.use_reference:
            if not ref_dir or not os.path.exists(ref_dir):
                raise ValueError(f"Reference flag is True, but invalid ref_dir provided: {ref_dir}")
            
            ref_spectra_list = []
            for root, dirs, files in os.walk(ref_dir):
                for file in files:
                    if file.lower().endswith('.csv') and 'summary' not in file.lower():
                        csv_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(csv_path)
                            
                            # Get the actual wavenumbers for the reference spectrum
                            if wavenumber_column in df.columns:
                                ref_wv = df[wavenumber_column].to_numpy(dtype=np.float32)
                            else:
                                logger.warning(f"Skipping {file}: Missing '{wavenumber_column}'")
                                continue
                                
                            drop_cols = ['Raman shift'] + [c for c in df.columns if c.startswith("St.Dev") or c.startswith("average")]
                            df = df.drop(columns=drop_cols, errors='ignore')
                            
                            intensity_columns = [c for c in df.columns if c != wavenumber_column and not c.lower().startswith("unnamed")]
                            
                            for col in intensity_columns:
                                raw_ref_intensity = df[col].to_numpy(dtype=np.float32)
                                
                                # Interpolate the reference intensity to perfectly match the 1011 mixture wavenumbers
                                interpolator = interp1d(
                                    ref_wv, 
                                    raw_ref_intensity, 
                                    kind='linear', 
                                    bounds_error=False, 
                                    fill_value="extrapolate"
                                )
                                # self.wavenumbers holds the 1011 mixture wavenumbers
                                resampled_intensity = interpolator(self.wavenumbers) 
                                
                                ref_spectra_list.append(resampled_intensity.astype(np.float32))
                                
                        except Exception as e:
                            logger.warning(f"Error loading reference {csv_path}: {e}")
            
            if not ref_spectra_list:
                raise ValueError(f"No valid reference spectra found in {ref_dir}")
                
            self.ref_samples = np.array(ref_spectra_list, dtype=np.float32)
            logger.info(f"Loaded {len(self.ref_samples)} pure reference spectra (Resized to {self.ref_samples.shape[1]} points)")
        else:
            self.ref_samples = None

        # -----------------------------
        # Baseline correction + smoothing
        # -----------------------------
        processed = []
        for s in self.samples:
            try:
                s_corr = airPLS_sparse(s, lam=1e5, porder=1, itermax=15, use_savgol=True)
                s_smooth = whittaker_smooth_sparse(
                    s_corr, w=np.ones_like(s_corr), lam=1e3, diff_order=2
                )
                processed.append(s_smooth)
            except Exception as e:
                logger.warning(f"Error processing spectrum: {e}")
                processed.append(s)  # Use original if processing fails
        
        self.samples = np.array(processed, dtype=np.float32)
        if self.use_reference:
            ref_processed = []
            for s in self.ref_samples:
                try:
                    s_corr = airPLS_sparse(s, lam=1e5, porder=1, itermax=15, use_savgol=True)
                    s_smooth = whittaker_smooth_sparse(
                        s_corr, w=np.ones_like(s_corr), lam=1e3, diff_order=2
                    )
                    ref_processed.append(s_smooth)
                except Exception as e:
                    logger.warning(f"Error processing spectrum: {e}")
                    ref_processed.append(s) 
            self.ref_samples = np.array(ref_processed, dtype=np.float32)
        
        # -----------------------------
        # Preprocessing
        # -----------------------------
        if preprocess == "z_score":
            self.samples = np.array([z_score(s) for s in self.samples])
        elif preprocess == "normalization":
            self.samples = np.array([normalization(s) for s in self.samples])
        elif preprocess == "vector":
            return vector_normalization(self.samples)
        if self.use_reference:
            if preprocess == "z_score":
                self.ref_samples = np.array([z_score(s) for s in self.ref_samples])
            elif preprocess == "normalization":
                self.ref_samples = np.array([normalization(s) for s in self.ref_samples])
            elif preprocess == "vector":
                return vector_normalization(self.ref_samples)

        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

        # -----------------------------
        # Build Pairs (Siamese Logic)
        # -----------------------------
        if self.use_reference:
            all_pairs = []
            all_targets = []
            
            for i, query_spec in enumerate(self.samples):
                label = self.labels[i]
                for ref_spec in self.ref_samples:
                    all_pairs.append((query_spec, ref_spec))
                    all_targets.append(label)
                    
            self.pairs = all_pairs
            self.targets = all_targets
            logger.info(f"Built {len(self.pairs)} query-reference pairs.")
        else:
            logger.info(f"Loaded {len(self.samples)} samples (no pairing).")
            
        # ==========================================
        # Dynamically log the active bin cutoffs
        # ==========================================
        active_bins = self.concentration_bins if self.concentration_bins is not None else [0.05, 0.5, 3.75, 15.0]
        
        # Build the dynamic log string
        log_parts = [f"L0 <= {active_bins[0]}"]
        for idx in range(len(active_bins) - 1):
            log_parts.append(f"{active_bins[idx]} < L{idx+1} <= {active_bins[idx+1]}")
        log_parts.append(f"L{len(active_bins)} > {active_bins[-1]}")
        
        bin_log_str = " | ".join(log_parts)
        logger.info(f"Active Bin Cutoffs: {bin_log_str}")
        logger.info(f"Labels: {set(self.labels)}")
        logger.info(f"Label distribution: {np.bincount(self.labels)}")

        # Establish slicing parameters once
        if self.sliced:
            first_sample = np.zeros(shape=(2, self.samples[0].shape[0]))
            self.window_size = int(first_sample.shape[-1] * self.window_ratio)
            self.window_num = slicer(first_sample, window_size=self.window_size,
                                     stride=self.stride, dilation=self.dilation).shape[1]

    
    def __len__(self):
        base_len = len(self.pairs) if self.use_reference else len(self.samples)
        if self.sliced:
            return base_len * self.window_num
        return base_len
    
    def __getitem__(self, idx):
        if self.sliced:
            origin_idx, win_idx = idx // self.window_num, idx % self.window_num
        else:
            origin_idx = idx

        if self.use_reference:
            # Siamese Return Mode
            query_spec, ref_spec = self.pairs[origin_idx]
            label = self.targets[origin_idx]
            
            if self.augment:
                query_spec = aug.random_augment(query_spec)
                
            x1 = np.stack([self.wavenumbers, query_spec], axis=0).astype(np.float32)
            x2 = np.stack([self.wavenumbers, ref_spec], axis=0).astype(np.float32)
            
            if self.sliced:
                x1 = slicer(x1, window_size=self.window_size, stride=self.stride, dilation=self.dilation)[:, win_idx, :]
                x2 = slicer(x2, window_size=self.window_size, stride=self.stride, dilation=self.dilation)[:, win_idx, :]

            return {
                "x": torch.tensor(x1, dtype=torch.float32),                     
                "ref": torch.tensor(x2, dtype=torch.float32),
                "y": torch.tensor(label, dtype=torch.long)
            }
        else:
            # Standard Return Mode
            spectrum = self.samples[origin_idx]
            label = self.labels[origin_idx]
            
            if self.augment:
                spectrum = aug.random_augment(spectrum)
                
            x = np.stack([self.wavenumbers, spectrum], axis=0).astype(np.float32)        
            
            if self.sliced:
                x = slicer(x, window_size=self.window_size, stride=self.stride, dilation=self.dilation)[:, win_idx, :]

            return {
                "x": torch.tensor(x, dtype=torch.float32),                     
                "y": torch.tensor(label, dtype=torch.long)
            }


class GBLConcentrationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        preprocess: str = "z_score",        
        val_split: float = 0.1,
        test_split: float = 0.1,
        augment: bool = True,
        sliced: bool = False,
        window_ratio: float = 1.0,
        stride: int = 1,
        dilation: int = 1,
        wavenumber_column: str = "Raman shift",
        use_reference: bool = False,
        ref_dir: Optional[str] = None,
        concentration_bins: Optional[List[float]] = None  
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preprocess = preprocess
        self.sliced = sliced
        self.window_ratio = window_ratio
        self.stride = stride
        self.dilation = dilation
        self.augment = augment
        self.val_split = val_split
        self.test_split = test_split
        self.wavenumber_column = wavenumber_column
        self.use_reference = use_reference
        self.ref_dir = ref_dir
        self.concentration_bins = concentration_bins
        
    def setup(self, stage=None):
        dataset = GBLConcentrationDataset(
            data_dir=self.data_dir,
            preprocess=self.preprocess,
            augment=self.augment,
            sliced=self.sliced,
            window_ratio=self.window_ratio,
            stride=self.stride,
            dilation=self.dilation,
            wavenumber_column=self.wavenumber_column,
            use_reference=self.use_reference,
            ref_dir=self.ref_dir,
            concentration_bins=self.concentration_bins
        )
        
        # Calculate split sizes
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
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