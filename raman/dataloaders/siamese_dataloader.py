from typing import Optional
from torch.utils.data import Dataset, DataLoader
from itertools import combinations, product
import lightning as L
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)

class SiameseAllPairsDataset(Dataset):
    def __init__(self, base_dataset: Dataset, subsampling_ratio: float = 0.2):
        self.base_dataset = base_dataset
        self.label_to_indices = {}
        self.subsampling_ratio = subsampling_ratio

        # Build label → list of indices mapping
        for idx in range(len(base_dataset)):
            label = base_dataset[idx]['y']
            self.label_to_indices.setdefault(label, []).append(idx)

        print(f"Base dataset size: {len(base_dataset)}")
        for label, indices in self.label_to_indices.items():
            print(f"Label {label}: {len(indices)} items")

        # Generate positive and negative pairs
        positive_pairs = []
        negative_pairs = []

        # Positive pairs: within the same class
        for label, indices in self.label_to_indices.items():
            for i, j in combinations(indices, 2):
                positive_pairs.append(((i, j), 1))

        # Negative pairs: between different classes
        labels = list(self.label_to_indices.keys())
        print("Found labels:", labels)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                label_i = labels[i]
                label_j = labels[j]
                for idx_i in self.label_to_indices[label_i]:
                    for idx_j in self.label_to_indices[label_j]:
                        negative_pairs.append(((idx_i, idx_j), 0))

        # Balance the dataset
        num_pos = len(positive_pairs)
        num_neg = len(negative_pairs)
        min_len = int(min(num_pos, num_neg)*self.subsampling_ratio)

        positive_pairs = random.sample(positive_pairs, min_len) if num_pos > min_len else positive_pairs
        negative_pairs = random.sample(negative_pairs, min_len) if num_neg > min_len else negative_pairs

        balanced_pairs = positive_pairs + negative_pairs
        random.shuffle(balanced_pairs)

        self.pairs = [pair for pair, _ in balanced_pairs]
        self.targets = [label for _, label in balanced_pairs]

        logger.info(f"Generated {len(self.pairs)} balanced pairs: {sum(self.targets)} positive and {len(self.pairs) - sum(self.targets)} negative.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        sample1 = self.base_dataset[idx1]["x"]
        sample2 = self.base_dataset[idx2]["x"]
        target = self.targets[idx]
        return {"x1": sample1, "x2": sample2, "y": target}


class SiameseDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_datamodule: L.LightningDataModule,
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
        subsampling_ratio: float = 0.2,
        test_split: float = 0.1,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.base_datamodule = base_datamodule
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subsampling_ratio = subsampling_ratio
        self.test_split = test_split
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        self.base_datamodule.setup(stage)
        self.train_dataset = SiameseAllPairsDataset(self.base_datamodule.train_dataset, self.subsampling_ratio)
        self.val_dataset = SiameseAllPairsDataset(self.base_datamodule.val_dataset, self.subsampling_ratio)
        self.test_dataset = SiameseAllPairsDataset(self.base_datamodule.test_dataset, self.subsampling_ratio)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )