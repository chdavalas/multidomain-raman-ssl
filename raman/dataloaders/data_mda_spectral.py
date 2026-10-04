from typing import Optional
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributed import is_available, is_initialized
from raman.dataloaders.scripts.global_wavenumbers import global_wavenumber_range
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler, Subset
import os
import torch
import logging
import numpy as np
import scipy.io
import lightning as L
import json

logger = logging.getLogger(__name__)
def use_ddp():
    return is_available() and is_initialized()
# This dataset is for the MDA Spectral Denoising task, where we have separate .mat files for training and testing inputs and outputs.
# Also the intensities are by default between 0 and 1, so no normalization is needed. The wavenumbers are also loaded from a separate file and should be normalized.
# We do not augment due to the nature of the task (denoising).

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

class MDADenoisingDataset(Dataset):
    def __init__(self, input_file, output_file, wavenumbers_file):
        mat_in = scipy.io.loadmat(input_file)
        mat_out = scipy.io.loadmat(output_file)
        
        # Dynamically find the key that contains 'Inputs' or 'Outputs'
        # This handles 'Train_Inputs', 'Test_Inputs', etc.
        in_key = [k for k in mat_in.keys() if 'Inputs' in k][0]
        out_key = [k for k in mat_out.keys() if 'Outputs' in k][0]
        
        self.inputs = mat_in[in_key].astype(np.float32)
        self.outputs = mat_out[out_key].astype(np.float32)
        
        # Load wavenumbers
        self.wavenumbers = np.loadtxt(wavenumbers_file, dtype=np.float32)
        self.wavenumbers = global_wv_min_max_scaling(self.wavenumbers)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # x: [Seq_Len, 2] -> (wavenumbers, noisy_intensity)
        x = np.stack((self.wavenumbers, self.inputs[idx]), axis=0)
        
        # y: [Seq_Len, 2] -> (wavenumbers, clean_intensity)
        y = np.stack((self.wavenumbers, self.outputs[idx]), axis=0)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y)
        }
        
class MDADenoisingDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_input_file: str = './datasets/MDA-Spectral/Train_Inputs.mat',
        train_output_file: str = './datasets/MDA-Spectral/Train_Outputs.mat',
        test_input_file: str = './datasets/MDA-Spectral/Test_Inputs.mat',
        test_output_file: str = './datasets/MDA-Spectral/Test_Outputs.mat',
        axis_file: str = './datasets/MDA-Spectral/axis.txt',
        val_split: float = 0.1
    ):
        super().__init__()
        
        self.train_input_file = train_input_file
        self.train_output_file = train_output_file
        self.test_input_file = test_input_file
        self.test_output_file = test_output_file
        self.axis_file = axis_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        # Handle Training and Validation (Split from Train files)
        if stage == "fit" or stage is None:
            full_train_ds = MDADenoisingDataset(
                self.train_input_file, 
                self.train_output_file, 
                self.axis_file
            )
            
            val_size = int(len(full_train_ds) * self.val_split)
            train_size = len(full_train_ds) - val_size
            
            self.train_ds, self.val_ds = random_split(
                full_train_ds, [train_size, val_size]
            )
            # Handle Testing (Loaded from separate Test files)
            print(f"MDA Denoising Train dataset size: {len(self.train_ds)}")
            print(f"MDA Denoising Validation dataset size: {len(self.val_ds)}")


        if stage == "test" or stage is None:
            self.test_ds = MDADenoisingDataset(
                self.test_input_file, 
                self.test_output_file, 
                self.axis_file
            )
            print(f"MDA Denoising Test dataset size: {len(self.test_ds)}")


        
    def train_dataloader(self):

        if use_ddp():
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self):

        if use_ddp():
            sampler = DistributedSampler(self.val_ds, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):

        if use_ddp():
            sampler = DistributedSampler(self.test_ds, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.val_dataloader()
