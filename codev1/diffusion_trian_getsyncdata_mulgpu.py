
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
from dataprepfunc import load_dataset
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection



batch_size = 16
epochs = 10
timesteps = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'



image_size=167
# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_loader = load_dataset('train_data.pt', batch_size=batch_size, shuffle=True)
test_loader = load_dataset('test_data.pt', batch_size=batch_size, shuffle=False)

class SimpleDenoiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, image_size * image_size)
        )
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t, y):
        y_emb = self.label_emb(y.float().view(-1, 1)).view(y.size(0), 1, image_size, image_size)
        x = torch.cat([x, y_emb], dim=1)
        return self.net(x)

# Improved Noise Scheduler
class NoiseScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]])

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alpha_cumprod = (1.0 / torch.sqrt(self.alpha_cumprod[t])).view(-1, 1, 1, 1)
        sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod[t] - 1).view(-1, 1, 1, 1)
        return sqrt_recip_alpha_cumprod * x_t - sqrt_recipm1_alpha_cumprod * noise

# Initialize
model = SimpleDenoiseModel().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

scheduler = NoiseScheduler(timesteps)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Training
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        y = y.long().to(device).squeeze()
        t = torch.randint(0, timesteps, (x.size(0),), device=device).long()

        noise = torch.randn_like(x)
        x_noisy = scheduler.q_sample(x, t, noise)
        noise_pred = model(x_noisy, t, y)

        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# Inference
model.eval()
@torch.no_grad()
def sample(model, scheduler, shape, y_label):
    img = torch.randn(shape, device=device)
    y_label = torch.full((shape[0],), y_label, device=device, dtype=torch.long)
    for t in reversed(range(scheduler.timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model(img, t_batch, y_label)
        x_start = scheduler.predict_start_from_noise(img, t_batch, noise_pred)
        if t > 0:
            noise = torch.randn_like(img)
            beta = scheduler.betas[t]
            alpha = scheduler.alphas[t]
            alpha_cumprod = scheduler.alpha_cumprod[t]
            alpha_cumprod_prev = scheduler.alpha_cumprod_prev[t]
            posterior_var = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            img = torch.sqrt(alpha_cumprod_prev).view(-1, 1, 1, 1) * x_start + torch.sqrt(posterior_var).view(-1, 1, 1, 1) * noise
        else:
            img = x_start
    # De-normalize output back to [0, 1]
    img = (img + 1) / 2
    img = torch.clamp(img, 0.0, 1.0)

    return img

# Generate and Plot
samples_0 = sample(model, scheduler, (4, 1, image_size, image_size), y_label=0).cpu()
samples_1 = sample(model, scheduler, (4, 1, image_size, image_size), y_label=1).cpu()

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i in range(4):
    axes[0, i].imshow(samples_0[i].squeeze(), cmap='coolwarm')
    axes[0, i].axis('off')
    axes[1, i].imshow(samples_1[i].squeeze(), cmap='coolwarm')
    axes[1, i].axis('off')
plt.show()



@torch.no_grad()
def generate_balanced_samples(model, scheduler, n_samples_per_label=400, save_path=r"/work/users/y/y/yyu1/892proj/generated_samples_10ep.mat"):
    model.eval()
    samples = []
    labels = []

    for label in [0, 1]:
        generated = []
        while len(generated) < n_samples_per_label:
            batch_size = min(32, n_samples_per_label - len(generated))
            imgs = sample(model, scheduler, (batch_size, 1, image_size, image_size), y_label=label).cpu()
            imgs = (imgs + 1) / 2  # De-normalize to [0, 1]
            imgs = torch.clamp(imgs, 0.0, 1.0)
            generated.append(imgs)
        generated = torch.cat(generated, dim=0)[:n_samples_per_label]
        samples.append(generated)
        labels.append(torch.full((n_samples_per_label,), label))

    samples = torch.cat(samples, dim=0).squeeze(1).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    scipy.io.savemat(save_path, {'samples': samples, 'labels': labels})
    print(f"Saved balanced samples to {save_path}")

# Function to conduct piecewise t-test and FDR correction
def piecewise_ttest_fdr(samples, labels, alpha=0.05):
    samples = torch.tensor(samples)
    labels = torch.tensor(labels)
    
    group0 = samples[labels == 0]
    group1 = samples[labels == 1]

    t_stat, p_vals = stats.ttest_ind(group0, group1, axis=0, equal_var=False)

    p_vals_flat = p_vals.flatten()
    reject, p_vals_corrected = fdrcorrection(p_vals_flat, alpha=alpha)

    significant_mask = reject.reshape(p_vals.shape)

    return significant_mask, p_vals.reshape(p_vals.shape), p_vals_corrected.reshape(p_vals.shape)
generate_balanced_samples(model=model,scheduler=scheduler)
# Example usage after generating samples:
data = scipy.io.loadmat(r"/work/users/y/y/yyu1/892proj/generated_samples_10ep.mat")
samples = data['samples']
labels = data['labels'].flatten()








