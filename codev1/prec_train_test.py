# file: run_experiment.py

import random
import numpy as np
import torch
import scipy.io
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
def load_dataset(file_path: str, batch_size: int = 32, shuffle: bool = True)-> DataLoader:
       
        data = scipy.io.loadmat(file_path)
        samples = data['samples']  # (N, 167, 167)
        labels = data['labels'].squeeze()  # (N,)

        samples = torch.tensor(samples, dtype=torch.float32).squeeze(1)  # (N, 1, 167, 167)
        labels = torch.tensor(labels, dtype=torch.float32)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        dataset = TensorDataset(samples, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader



def run_experiment(
    SEED,
    input_size,
    traindata='train_data.pt',
    testdata='test_data.pt',
    batch_size=64,
    latent_dim=120,
    num_latents=60,
    heads=1,
    num_self_attn_layers=4,
    dim_ca_ffw=512,
    dropout=0.1,
    num_iterations=4,
    num_epochs=40
):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader = load_dataset(traindata, batch_size=batch_size, shuffle=True)
    test_loader = load_dataset(testdata, batch_size=batch_size, shuffle=False)

    class PerceiverBinary(nn.Module):
        def __init__(self, input_size, latent_dim, num_latents, heads, 
                     num_self_attn_layers, dim_ca_ffw, dropout, num_iterations):
            super(PerceiverBinary, self).__init__()
            self.latent_dim = latent_dim
            self.num_latents = num_latents
            self.num_iterations = num_iterations
            self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)
            self.latent_pos_enc = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
            self.Wq = nn.Linear(latent_dim, latent_dim, bias=False)
            self.Wk = nn.Linear(input_size[1], latent_dim, bias=False)
            self.Wv = nn.Linear(input_size[1], latent_dim, bias=False)
            self.ln_q = nn.LayerNorm(latent_dim)
            self.ln_kv = nn.LayerNorm(latent_dim)
            self.ln_latents = nn.LayerNorm(latent_dim)
            self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=heads, batch_first=True)
            self.cross_attn_mlp = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, dim_ca_ffw),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ca_ffw, latent_dim),
                nn.Dropout(dropout)
            )
            self.self_attention_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=latent_dim, nhead=heads, dim_feedforward=1024, 
                    dropout=dropout, activation="gelu", norm_first=True
                ) for _ in range(num_self_attn_layers)
            ])
            self.fc_head_mean = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            batch_size = x.shape[0]
            latents = self.latents.expand(batch_size, -1, -1) + self.latent_pos_enc
            latents = self.ln_latents(latents)
            for _ in range(self.num_iterations):
                Q = self.ln_q(self.Wq(latents))
                K = self.ln_kv(self.Wk(x))
                V = self.ln_kv(self.Wv(x))
                attended_latents, _ = self.cross_attention(Q, K, V)
                latents = self.cross_attn_mlp(attended_latents) + attended_latents
                for layer in self.self_attention_layers:
                    latents = layer(latents)
            output = self.fc_head_mean(latents.mean(dim=1))
            return output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerceiverBinary(
        input_size=input_size,
        latent_dim=latent_dim,
        num_latents=num_latents,
        heads=heads,
        num_self_attn_layers=num_self_attn_layers,
        dim_ca_ffw=dim_ca_ffw,
        dropout=dropout,
        num_iterations=num_iterations
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    final_test_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = (outputs > 0.5).int()
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                predicted = (outputs > 0.5).int()
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_accuracy = 100 * correct_val / total_val
        final_test_acc = val_accuracy

        print(f"Seed {SEED} Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Test Acc: {val_accuracy:.2f}%")

        scheduler.step()

    return final_test_acc

# ----------------------------
# Run experiments with different seeds
# ----------------------------


