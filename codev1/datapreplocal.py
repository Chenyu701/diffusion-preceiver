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

# Configurations
batch_size = 16
epochs = 10
timesteps = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load dataset
mat_data = scipy.io.loadmat(r"C:\Users\Administrator\Desktop\brian network\modeling\hcp_yad\HCP_subcortical_CMData_destrieux")
all_id = mat_data['all_id'].flatten()
tensor_data = torch.tensor(mat_data['loaded_tensor_sub'], dtype=torch.float32)
tensor_data = tensor_data.permute(3, 2, 0, 1)
logical_matrix = tensor_data[:, 0, :, :]
logical_matrix = logical_matrix[:, :168, :168]

# Convert to DataFrame
df_mat = pd.DataFrame({'subject': all_id})
df_mat['logical_matrix'] = list(logical_matrix.numpy())

# Load CSV
dfy = pd.read_csv(r"C:\Users\Administrator\Desktop\brian network\modeling\hcp_yad\unrestricted.csv")
dfy.rename(columns={"Subject": "subject"}, inplace=True)
df_merged = dfy.merge(df_mat, on="subject", how="inner")
df_merged['Gender'] = df_merged['Gender'].map({'F': 0, 'M': 1})

# Prepare Dataset
X_all = torch.tensor(np.stack(df_merged['logical_matrix'].values), dtype=torch.float32)
num_matrices, _, _ = X_all.shape
for i in range(num_matrices):
    matrix = X_all[i, :, :]
    lower_triangular = torch.tril(matrix, diagonal=-1)
    upper_triangular = torch.triu(matrix, diagonal=1)
    recovered_matrix = lower_triangular + upper_triangular.T + upper_triangular
    X_all[i, :, :] = recovered_matrix
    
    # Apply logarithm (with epsilon) and adjust diagonal
epsilon = 1e-6
X_all = torch.log(X_all + epsilon)
for i in range(num_matrices):
    X_all[i, :, :] += torch.eye(167, device=X_all.device) * 9
    
    # Crop X_all to (N,150,150) to match model's expected input size
X_all = X_all[:, :, :]
X_all = X_all.unsqueeze(1)
# Normalize X_all to [-1, 1]
X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
X_all = X_all * 2 - 1

y_all = torch.tensor(df_merged['Gender'].values, dtype=torch.float32).view(-1, 1)

# Update image size automatically
image_size = X_all.shape[-1]

# Dataset Loader
dataset = TensorDataset(X_all, y_all)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 




# File: split_save_dataset.py

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

import torch
import scipy.io
from torch.utils.data import TensorDataset, random_split

def split_and_save_dataset(X_all: torch.Tensor, y_all: torch.Tensor, train_ratio: float = 0.8):

    torch.manual_seed(42)

    full_dataset = TensorDataset(X_all, y_all)
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Extract and stack tensors
    X_train, y_train = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    X_test, y_test = zip(*[test_dataset[i] for i in range(len(test_dataset))])

    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    # Convert to numpy
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Save as .mat
    scipy.io.savemat(r'C:\Users\Administrator\Desktop\brian network\892finalproj\train_data.mat', {'samples': X_train_np, 'labels': y_train_np})
    scipy.io.savemat(r'C:\Users\Administrator\Desktop\brian network\892finalproj\test_data.mat', {'samples': X_test_np, 'labels': y_test_np})

    print("Datasets saved: train_data.mat and test_data.mat")


# Example usage
split_and_save_dataset(X_all, y_all)




















