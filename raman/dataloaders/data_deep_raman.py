import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler

from torch.utils.data import Dataset, DataLoader
import lightning as L
import logging
import copy
import json
import csv
# import raman.dataloaders.augmentations as aug
from torch.distributed import is_available, is_initialized


import raman.dataloaders.scripts.augmentations as aug
import json
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
import os
logger = logging.getLogger(__name__)

def use_ddp():
    return is_available() and is_initialized()

def match_model_length(spectra, target_points=881, start=240, end=2000):
    """If spectra length ≠ target_points, interpolate to target length."""
    if spectra.shape[1] == target_points:
        return spectra
    num_points = spectra.shape[1]
    new_spectra = np.zeros((spectra.shape[0], target_points), dtype=np.float32)
    for i in range(spectra.shape[0]):
        new_spectra[i, :] = np.interp(
            np.linspace(start, end, target_points),
            np.linspace(start, end, num_points),
            spectra[i, :]
        )
    return new_spectra

def randomize(X, Y):
    indices = np.random.permutation(len(Y))
    return X[indices], Y[indices]

class DeepRamanTestDataset(Dataset):
    def __init__(self, datapath):
        # Load .npy files
        pure_spectra = np.load(os.path.join(datapath, "database_for_Liquid_and_powder_mixture.npy"))
        mix_spectra = np.load(os.path.join(datapath, "unknown_Liquid_and_powder_mixture.npy"))
        # Component information of database
        mix_spectra = np.load(os.path.join(datapath, "unknown_Liquid_and_powder_mixture.npy"))
        csv_path = os.path.join(datapath, "database_for_Liquid_and_powder_mixture.csv")
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            component_names = [row[0] for row in reader]

        spectrum_pure_sc =  copy.deepcopy(pure_spectra)
        spectrum_mix_sc = copy.deepcopy(mix_spectra)
        for i in range(mix_spectra.shape[0]):
            spectrum_mix_sc[i,:] = mix_spectra[i,:]/np.max(mix_spectra[i,:])
        for i in range(pure_spectra.shape[0]):
            spectrum_pure_sc[i,:] = pure_spectra[i,:]/np.max(pure_spectra[i,:])

        num_mix = spectrum_mix_sc.shape[0]
        num_pure = spectrum_pure_sc.shape[0]

        # Build pairs and store component names and mixture indices
        self.pairs = []
        for mix_idx in range(num_mix):
            for comp_idx in range(num_pure):
                mix_spec = spectrum_mix_sc[mix_idx]
                comp_spec = spectrum_pure_sc[comp_idx]
                comp_name = component_names[comp_idx]
                self.pairs.append((mix_spec, comp_spec, comp_name, mix_idx))
        
        self.wavenumbers = generate_wavenumbers(num_points=881)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)
        

    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        mix, comp, comp_name, mix_idx = self.pairs[idx]
        
        x1 =  np.float32(np.vstack((self.wavenumbers, mix)))
        x2 =  np.float32(np.vstack((self.wavenumbers, comp)))
        return {
            "x": x1,
            "ref": x2,
            "component_name": comp_name,
            "mixture_index": mix_idx
        }

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


def generate_wavenumbers(start=240, end=2000, num_points=881):
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
    
    logger.info("Used global wavenumbers min max scalling!")
    return scaled_data
    
    
class DeepRamanTestDataset(Dataset):
    def __init__(self, pure_file=None, mix_file=None, dbcoms_file=None, 
                 mix_file_only=None, pure_from_mix_first_n=None, dbcoms_list=None):
        """
        Supports:
        (A) Standard DeepRaman evaluation: use pure_file, mix_file, dbcoms_file
        (B) Compact evaluation: use mix_file_only + pure_from_mix_first_n + dbcoms_list
        """
        
        # --- Case A: standard DeepRaman setup ---
        if pure_file is not None and mix_file is not None and dbcoms_file is not None:
            pure_spectra = np.load(pure_file)
            mix_spectra = np.load(mix_file)
            with open(dbcoms_file, encoding="utf-8") as f:
                reader = csv.reader(f)
                component_names = [row[0] for row in reader]

        # --- Case B: new compact setup ---
        elif mix_file_only is not None and pure_from_mix_first_n is not None and dbcoms_list is not None:
            all_spectra = np.load(mix_file_only)
            pure_spectra = all_spectra[:pure_from_mix_first_n]
            mix_spectra = all_spectra[pure_from_mix_first_n:]
            component_names = dbcoms_list
        else:
            raise ValueError("Invalid initialization: must provide either "
                             "(pure_file, mix_file, dbcoms_file) or "
                             "(mix_file_only, pure_from_mix_first_n, dbcoms_list).")

        # --- Normalize spectra ---
        spectrum_pure_sc = pure_spectra / np.max(pure_spectra, axis=1, keepdims=True)
        spectrum_mix_sc = mix_spectra / np.max(mix_spectra, axis=1, keepdims=True)

        # Resample if needed
        spectrum_pure_sc = match_model_length(spectrum_pure_sc, target_points=881)
        spectrum_mix_sc = match_model_length(spectrum_mix_sc, target_points=881)


        num_mix = spectrum_mix_sc.shape[0]
        num_pure = spectrum_pure_sc.shape[0]

        # --- Build all (mix, pure) pairs ---
        self.pairs = []
        for mix_idx in range(num_mix):
            for comp_idx in range(num_pure):
                mix_spec = spectrum_mix_sc[mix_idx]
                comp_spec = spectrum_pure_sc[comp_idx]
                comp_name = component_names[comp_idx]
                self.pairs.append((mix_spec, comp_spec, comp_name, mix_idx))

        self.wavenumbers = generate_wavenumbers(num_points=881)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mix, comp, comp_name, mix_idx = self.pairs[idx]
        x1 = np.float32(np.vstack((self.wavenumbers, mix)))
        x2 = np.float32(np.vstack((self.wavenumbers, comp)))
        return {
            "x": x1,
            "ref": x2,
            "component_name": comp_name,
            "mixture_index": mix_idx,
        }

class DeepRamanDataset(Dataset):
    def __init__(self, datapath, split="train", val_split=0.1, preprocess: str = "z_score", augment: bool = True):
        # Load data
        X = np.load(os.path.join(datapath, "training_X.npy"))  # shape: (N, 2, channels, 1)
        Y = np.load(os.path.join(datapath, "training_Y.npy"))  # shape: (N, 2)
    
        # Flatten pair dimension
        X = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3])
        Y = Y.reshape(Y.shape[0]*Y.shape[1],1)

        Y = Y.astype(int)

        X, Y = randomize(X, Y)
        
        X = X.astype(np.float32)

        # Split
        split_idx = int((1 - val_split) * len(Y))
        if split == "train":
            self.X = X[:split_idx]
            self.Y = Y[:split_idx]
        else:
            self.X = X[split_idx:]
            self.Y = Y[split_idx:]
            
        if preprocess == "z_score":
            self.X = z_score(self.X)
        elif preprocess == "normalization":
            self.X = normalization(self.X)
        elif preprocess == "vector":
            self.X = vector_normalization(self.X)
        else:
            logger.info("No preprocess was used.")   
            
        self.wavenumbers = generate_wavenumbers(num_points=881)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)
        self.augment = augment

        logger.info(f"Loaded DeepRamanDataset with {len(self.X)} fixed pairs: "
                    f"{np.sum(self.Y)} positive and {len(self.Y) - np.sum(self.Y)} negative.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx, 0]
        x2 = self.X[idx, 1]

        if self.augment:
            x1 = aug.random_augment(x1)
            x2 = aug.random_augment(x2)

        sample1 = np.vstack((self.wavenumbers, x1))
        sample2 = np.vstack((self.wavenumbers, x2))

        y = self.Y[idx, 0]
        return {"x": sample1, "ref": sample2, "y": y}


class DeepRamanDataModule(L.LightningDataModule):
    def __init__(
        self, 
        datapath, 
        batch_size: int = 64,
        num_workers: int = 1,
        val_split: float = 0.1,
        pin_memory: bool = True,
        preprocess: str = "z_score",
        augment: bool = True,
        evaluation: dict = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datapath"])
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.preprocess = preprocess
        self.evaluation = evaluation or {}


    def setup(self, stage=None):
        self.train_dataset = DeepRamanDataset(
            self.datapath, split="train",
            val_split=self.hparams.val_split,
            preprocess=self.hparams.preprocess,
            augment=self.hparams.augment
        )
        
        self.val_dataset = DeepRamanDataset(
            self.datapath, split="val",
            val_split=self.hparams.val_split,
            preprocess=self.hparams.preprocess,
            augment=self.hparams.augment
        )
        
        # Test dataset comes from config
        eval_cfg = self.evaluation
        # === Case A: Standard DeepRaman files ===
        if all(k in eval_cfg for k in ["pure_file", "mix_file", "dbcoms_file"]):
            self.test_dataset = DeepRamanTestDataset(
                pure_file=eval_cfg["pure_file"],
                mix_file=eval_cfg["mix_file"],
                dbcoms_file=eval_cfg["dbcoms_file"]
            )

        # === Case B: New compact dataset ===
        elif all(k in eval_cfg for k in ["mix_file_only", "pure_from_mix_first_n", "dbcoms_list"]):
            self.test_dataset = DeepRamanTestDataset(
                mix_file_only=eval_cfg["mix_file_only"],
                pure_from_mix_first_n=eval_cfg["pure_from_mix_first_n"],
                dbcoms_list=eval_cfg["dbcoms_list"]
            )

        else:
            raise ValueError("Invalid evaluation configuration in config.yaml. "
                            "Must include either DeepRaman-style or compact evaluation keys.")
        
        print(f"DeepRaman Train dataset size: {len(self.train_dataset)}")
        print(f"DeepRaman Validation dataset size: {len(self.val_dataset)}")
        print(f"DeepRaman Test dataset size: {len(self.test_dataset)}")

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
