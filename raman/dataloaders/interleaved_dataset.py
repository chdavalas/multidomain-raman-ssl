from typing import List, Optional
from torch.utils.data import DataLoader, Sampler
from itertools import cycle
import lightning as L
import logging
import torch

logger = logging.getLogger(__name__)

class InterleavedDataLoader:
    def __init__(self, dataloaders: List[DataLoader]):
        self.dataloaders = dataloaders
        self.max_len = max([len(dataloader) for dataloader in dataloaders])
        self.iterators = [cycle(dataloader) for dataloader in dataloaders]
        self.index = 0

    def __iter__(self):
        self.iterators = [cycle(dataloader) for dataloader in self.dataloaders]
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        batch = next(self.iterators[self.index % len(self.dataloaders)])
        self.index += 1
        return batch

    def __len__(self):
        return self.max_len * len(self.dataloaders)
        
class InterleavedModule(L.LightningDataModule):
    def __init__(
        self,
        datamodules: List[L.LightningDataModule],
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        logger.info(f"\nCurrent stage: {stage}")
        
        # Log the names of the loaded DataModules
        logger.info("Loaded DataModules:")
        for i, datamodule in enumerate(self.datamodules):
            print(f"  DataModule {i + 1}: {datamodule.__class__.__name__}")
        
        # Setup each DataModule
        for datamodule in self.datamodules:
            datamodule.setup(stage)

    def train_dataloader(self):
        # Create an InterleavedDataLoader for the train splits
        train_dataloaders = [datamodule.train_dataloader() for datamodule in self.datamodules]
        return InterleavedDataLoader(train_dataloaders)

    def val_dataloader(self):
        # Create an InterleavedDataLoader for the val splits
        val_dataloaders = [datamodule.val_dataloader() for datamodule in self.datamodules]
        return InterleavedDataLoader(val_dataloaders)

    def test_dataloader(self):
        # Create an InterleavedDataLoader for the test splits
        test_dataloaders = [datamodule.test_dataloader() for datamodule in self.datamodules]
        return InterleavedDataLoader(test_dataloaders)

    def plot_dataloader(self):
        # Create an InterleavedDataLoader for the predict splits
        predict_dataloaders = [datamodule.test_dataloader() for datamodule in self.datamodules]
        return InterleavedDataLoader(predict_dataloaders)    
