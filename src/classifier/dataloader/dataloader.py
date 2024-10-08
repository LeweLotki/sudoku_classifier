import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
from .puzzle_dataset import PuzzleDataset

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(csv_path, vectorization='tf-idf',batch_size=32, max_features=10000, train_ratio=0.7, val_ratio=0.15, shuffle=True):
    dataset = PuzzleDataset(csv_path, max_features=max_features,
    vectorization=vectorization)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=train_ratio, val_ratio=val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


