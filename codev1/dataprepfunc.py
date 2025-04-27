import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import random
from sklearn.model_selection import KFold



# File: split_save_dataset.py

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

# Assume X_all, y_all already prepared as in your code

def split_and_save_dataset(X_all: torch.Tensor, y_all: torch.Tensor, train_ratio: float = 0.8):
    torch.manual_seed(42)
    full_dataset = TensorDataset(X_all, y_all)
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    X_train, y_train = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    X_test, y_test = zip(*[test_dataset[i] for i in range(len(test_dataset))])

    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    torch.save({'X': X_train, 'y': y_train}, 'train_data.pt')
    torch.save({'X': X_test, 'y': y_test}, 'test_data.pt')

    print("Datasets saved: train_data.pt and test_data.pt")

def load_dataset(filepath: str, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    data = torch.load(filepath)
    dataset = TensorDataset(data['X'], data['y'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)