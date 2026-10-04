from typing import List, Optional
from torch.utils.data import DataLoader, Sampler
from itertools import cycle
import lightning as L
import logging
import torch

logger = logging.getLogger(__name__)

class MultiTaskInterleavedDataLoader:
    def __init__(self, loaders_with_tasks):
        """
        loaders_with_tasks: list of dicts:
            [{"loader": DataLoader, "task": str}, ...]
        """
        self.loaders_with_tasks = loaders_with_tasks
        self.max_len = max(len(item["loader"]) for item in loaders_with_tasks)
        self.iterators = [
            cycle(item["loader"]) for item in loaders_with_tasks
        ]
        self.tasks = [
            item["task"] for item in loaders_with_tasks
        ]
        self.index = 0

    def __iter__(self):
        self.iterators = [
            cycle(item["loader"]) for item in self.loaders_with_tasks
        ]
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration

        i = self.index % len(self.iterators)

        batch = next(self.iterators[i])
        batch["task"] = self.tasks[i] 
        self.index += 1
        return batch

    def __len__(self):
        return self.max_len * len(self.loaders_with_tasks)
        
class MultiTaskDataModule(L.LightningDataModule):

    def __init__(self, datasets_config):
        """
        datasets_config example:
        [
            {"datamodule": dm_a, "task": "denoising"},
            {"datamodule": dm_b, "task": "pretraining"}
        ]
        """
        super().__init__()
        self.datasets_config = datasets_config

    def setup(self, stage=None):
        logger.info(f"\nCurrent stage: {stage}")

        logger.info("Loaded DataModules:")
        for i, item in enumerate(self.datasets_config):
            print("DataModule ", i + 1,":", item["datamodule"], "| Task:", item["task"])
            item["datamodule"].setup(stage)


    def train_dataloader(self):

        loaders_with_tasks = []

        for item in self.datasets_config:
            loaders_with_tasks.append({
                "loader": item["datamodule"].train_dataloader(),
                "task": item["task"]
            })

        return MultiTaskInterleavedDataLoader(loaders_with_tasks)
    
    def val_dataloader(self):

        loaders_with_tasks = []

        for item in self.datasets_config:
            loaders_with_tasks.append({
                "loader": item["datamodule"].val_dataloader(),
                "task": item["task"]
            })

        return MultiTaskInterleavedDataLoader(loaders_with_tasks)
    
    
    def test_dataloader(self):

        loaders_with_tasks = []

        for item in self.datasets_config:
            loaders_with_tasks.append({
                "loader": item["datamodule"].test_dataloader(),
                "task": item["task"]
            })

        return MultiTaskInterleavedDataLoader(loaders_with_tasks)
