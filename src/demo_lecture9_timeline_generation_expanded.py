import math
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. DATASETS (Synthetic & NASA C-MAPSS)
# ==========================================

def generate_synthetic_sensor_data(num_samples=1000, channels=3, length=64):
    """Simulates multi-channel engineering telemetry."""
    t = torch.linspace(0, 10, length)
    dataset = []
    for _ in range(num_samples):
        freq = torch.rand(1) * 2.0 + 1.0
        phase = torch.rand(1) * math.pi
        c1 = torch.sin(freq * t + phase) + 0.1 * torch.randn(length)
        c2 = torch.cos(freq * 0.5 * t) * 2.0 + 0.05 * torch.randn(length)
        c3 = (t / 10.0) + 0.2 * torch.randn(length)
        sample = torch.stack([c1, c2, c3], dim=0)
        dataset.append(sample)
    return torch.stack(dataset, dim=0)

def load_cmapss_data(file_path="train_FD001.txt", seq_len=64, stride=10):
    """
    Loads NASA C-MAPSS dataset and slices into sliding windows.
    Expects space-separated text file with unit, cycle, 3 settings, 21 sensors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"C-MAPSS file '{file_path}' not found. Please place train_FD001.txt in the working directory "
            "or switch DATASET_CHOICE to 'synthetic'."
        )
    
    # C-MAPSS columns: id, cycle, setting1, setting2, setting3, s1...s21
    col_names = ['unit', 'cycle', 's1', 's2', 's3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names, usecols=range(26))
    
    # Select subset of active/varying sensors (e.g., sensors 2, 3, 4, 7, 8, 11, 12, 13, 14, 15, 17, 20, 21)
    sensor_cols = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 11, 12, 13, 14, 15]]
    
    windows = []
    for unit_id, group in df.groupby('unit'):
        sensor_data = group[sensor_cols].values
        # Z-score normalization per engine unit
        sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-5)
        
        # Sliding window extraction [Channels, TimeSteps]
        for start in range(0, len(sensor_data) - seq_len, stride):
            window = sensor_data[start:start + seq_len] # Shape: [seq_len, Channels]
            windows.append(torch.tensor(window.T, dtype=torch.float32)) # Shape: [Channels, seq_len]
            
    return torch.stack(windows, dim=0)


# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================

# --- A. Diffusion Model Components ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class Simple1DResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t):
        h = self.conv1(F.gelu(x))
        h = h + self.time_mlp(F.gelu(t)).unsqueeze(-1)
        h = self.conv2(F.gelu(h))
        return h + self.res_conv(x)

class TimeSeries1DDenoiser(nn.Module):
    def __init__(self, in_channels, base_dim=32, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_dim), nn.Linear(time_dim, time_dim), nn.GELU())
        self.init_conv = nn.Conv1d(in_channels, base_dim, 3, padding=1)
        self.block1 = Simple1DResBlock(base_dim, base_dim * 2, time_dim)
        self.down1 = nn.Conv1d(base_dim * 2, base_dim * 2, 4, stride=2, padding=1)
        self.bottleneck = Simple1DResBlock(base_dim * 2, base_dim * 4, time_dim)
        self.up1 = nn.ConvTranspose1d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.block2 = Simple1DResBlock(base_dim * 4, base_dim, time_dim)
        self.final_conv = nn.Conv1d(base_dim, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h1 = self.block1(self.init_conv(x), t_emb)
        bn = self.bottleneck(self.down1(h1), t_emb)
        up = torch.cat([self.up1(bn), h1], dim=1)
        return self.final_conv(F.gelu(self.block2(up, t_emb)))


# --- B. Encoder-Decoder Autoencoder (AE) ---
class TimeSeriesAE(nn.Module):
    def __init__(self, in_channels, seq_len, latent_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * (seq_len // 4), latent_dim)
        )
        self.dec_input = nn.Linear(latent_dim, 64 * (seq_len // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, in_channels, 4, stride=2, padding=1)
        )
    def forward(self, x):
        b, c, l = x.shape
        z = self.encoder(x)
        h = self.dec_input(z).view(b, 64, l // 4)
        return self.decoder(h)


# --- C. Variational Autoencoder (VAE) ---
class TimeSeriesVAE(nn.Module):
    def __init__(self, in_channels, seq_len, latent_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.enc_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten()
        )
        flat_dim = 64 * (seq_len // 4)
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)
        self.dec_input = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, in_channels, 4, stride=2, padding=1)
        )
    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_var(h)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    def decode(self, z):
        b = z.shape[0]
        h = self.dec_input(z).view(b, 64, self.seq_len // 4)
        return self.decoder(h)
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DResBlockAE(nn.Module):
    """A residual block for the Autoencoder (no time embedding needed)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.gelu(x))
        h = self.conv2(F.gelu(h))
        return h + self.res_conv(x)


class TimeSeriesUnetAE(nn.Module):
    def __init__(self, in_channels, base_dim=32):
        super().__init__()
        
        self.init_conv = nn.Conv1d(in_channels, base_dim, 3, padding=1)
        
        # --- Encoder Path ---
        self.block1 = Simple1DResBlockAE(base_dim, base_dim * 2)       # Out: 64
        self.down1 = nn.Conv1d(base_dim * 2, base_dim * 2, 4, stride=2, padding=1)
        
        self.block2 = Simple1DResBlockAE(base_dim * 2, base_dim * 4)   # Out: 128
        self.down2 = nn.Conv1d(base_dim * 4, base_dim * 4, 4, stride=2, padding=1)
        
        # --- Bottleneck ---
        self.bottleneck = Simple1DResBlockAE(base_dim * 4, base_dim * 8) # Out: 256
        
        # --- Decoder Path ---
        self.up2 = nn.ConvTranspose1d(base_dim * 8, base_dim * 4, 4, stride=2, padding=1) # Out: 128
        # cat2 = 128 (up2) + 128 (h2) = 256 channels
        self.decode2 = Simple1DResBlockAE(base_dim * 8, base_dim * 2) # Out: 64
        
        self.up1 = nn.ConvTranspose1d(base_dim * 2, base_dim, 4, stride=2, padding=1)     # Out: 32
        # cat1 = 32 (up1) + 64 (h1) = 96 channels
        self.decode1 = Simple1DResBlockAE(base_dim + base_dim * 2, base_dim)             # Out: 32
        
        # Final reconstruction layer back to original channels
        self.final_conv = nn.Conv1d(base_dim, in_channels, 3, padding=1)

    def forward(self, x):
        x0 = self.init_conv(x)
        
        # Encoder
        h1 = self.block1(x0)           # 64 channels (Saved for skip 1)
        x_d1 = self.down1(h1)
        
        h2 = self.block2(x_d1)         # 128 channels (Saved for skip 2)
        x_d2 = self.down2(h2)
        
        # Bottleneck
        bn = self.bottleneck(x_d2)     # 256 channels
        
        # Decoder Level 2
        u2 = self.up2(bn)              # 128 channels
        cat2 = torch.cat([u2, h2], dim=1) # 128 + 128 = 256 channels
        d2 = self.decode2(cat2)        # 64 channels
        
        # Decoder Level 1
        u1 = self.up1(d2)              # 32 channels
        cat1 = torch.cat([u1, h1], dim=1) # 32 + 64 = 96 channels
        d1 = self.decode1(cat1)        # 32 channels
        
        return self.final_conv(F.gelu(d1))
    
class SlimTimeSeriesUnetAE(nn.Module):
    def __init__(self, in_channels=8, base_dim=8):
        super().__init__()
        self.init_conv = nn.Conv1d(in_channels, base_dim, 3, padding=1)
        
        # --- Encoder (128 -> 64 -> 32 -> 16) ---
        self.block1 = Simple1DResBlockAE(base_dim, base_dim * 2)       # 16 ch, len 128
        self.down1 = nn.Conv1d(base_dim * 2, base_dim * 2, 4, stride=2, padding=1)
        
        self.block2 = Simple1DResBlockAE(base_dim * 2, base_dim * 4)   # 32 ch, len 64
        self.down2 = nn.Conv1d(base_dim * 4, base_dim * 4, 4, stride=2, padding=1)
        
        self.block3 = Simple1DResBlockAE(base_dim * 4, base_dim * 8)   # 64 ch, len 32
        self.down3 = nn.Conv1d(base_dim * 8, base_dim * 8, 4, stride=2, padding=1)
        
        # --- Bottleneck (Length 16, 64 channels = 1,024 total values) ---
        self.bottleneck = Simple1DResBlockAE(base_dim * 8, base_dim * 8)
        
        # --- Decoder (Upscaling back up) ---
        self.up3 = nn.ConvTranspose1d(base_dim * 8, base_dim * 4, 4, stride=2, padding=1)
        self.decode3 = Simple1DResBlockAE((base_dim * 8) + (base_dim * 4), base_dim * 4)
        
        self.up2 = nn.ConvTranspose1d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.decode2 = Simple1DResBlockAE((base_dim * 4) + (base_dim * 2), base_dim * 2)
        
        self.up1 = nn.ConvTranspose1d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.decode1 = Simple1DResBlockAE((base_dim * 2) + base_dim, base_dim)
        
        self.final_conv = nn.Conv1d(base_dim, in_channels, 3, padding=1)

    def forward(self, x):
        x0 = self.init_conv(x)
        
        h1 = self.block1(x0)
        x_d1 = self.down1(h1)
        
        h2 = self.block2(x_d1)
        x_d2 = self.down2(h2)
        
        h3 = self.block3(x_d2)
        x_d3 = self.down3(h3)
        
        bn = self.bottleneck(x_d3)  # Shape: [Batch, 64, 16] -> 1,024 values
        
        u3 = self.up3(bn)
        d3 = self.decode3(torch.cat([u3, h3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.decode2(torch.cat([u2, h2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.decode1(torch.cat([u1, h1], dim=1))
        
        return self.final_conv(F.gelu(d1))
        
# ==========================================
# 3. TRAINING & INFERENCE ROUTINES
# ==========================================

TIMESTEPS = 100
beta = torch.linspace(0.0001, 0.02, TIMESTEPS)
alpha = 1.0 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample_diffusion(model, shape, device):
    model.eval()
    x = torch.randn(shape, device=device)
    for i in reversed(range(TIMESTEPS)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        pred_noise = model(x, t)
        a_t, a_cum_t, b_t = alpha[i], alpha_cumprod[i], beta[i]
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = (1.0 / torch.sqrt(a_t)) * (x - (b_t / torch.sqrt(1.0 - a_cum_t)) * pred_noise) + torch.sqrt(b_t) * noise
    return x

@torch.no_grad()
def sample_vae(model, num_samples, in_channels, seq_len, device, latent_dim=32):
    model.eval()
    z = torch.randn(num_samples, latent_dim, device=device)
    return model.decode(z)


# ==========================================
# 4. MAIN EXECUTION & VISUALIZATION
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CONFIGURATION TOGGLES ---
    DATASET_CHOICE = 'synthetic' # Options: 'synthetic' or 'cmapss' or 'unet'
    MODEL_CHOICE = 'unet_slim'   # Options: 'diffusion', 'ae', 'unet', 'unet_slim' or 'vae'
    SEQ_LEN = 64
    EPOCHS = 50

    # Step 1: Load Dataset
    if DATASET_CHOICE == 'synthetic':
        print("Using Synthetic Dataset...")
        raw_data = generate_synthetic_sensor_data(num_samples=1000, channels=3, length=SEQ_LEN)
    elif DATASET_CHOICE == 'cmapss':
        print("Loading NASA C-MAPSS Dataset (train_FD001.txt)...")
        raw_data = load_cmapss_data(r'..\data\train_FD001.txt', seq_len=SEQ_LEN)
        
    in_channels = raw_data.shape[1]
    dataloader = DataLoader(raw_data, batch_size=32, shuffle=True)

    # Step 2: Initialize Model
    if MODEL_CHOICE == 'diffusion':
        model = TimeSeries1DDenoiser(in_channels=in_channels).to(device)
    elif MODEL_CHOICE == 'ae':
        model = TimeSeriesAE(in_channels=in_channels, seq_len=SEQ_LEN).to(device)
    elif MODEL_CHOICE == 'vae':
        model = TimeSeriesVAE(in_channels=in_channels, seq_len=SEQ_LEN).to(device)
    elif MODEL_CHOICE == 'unet':
        model = TimeSeriesUnetAE(in_channels=in_channels).to(device)
    elif MODEL_CHOICE == 'unet_slim':
        model = SlimTimeSeriesUnetAE(in_channels=in_channels).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Step 3: Training Loop
    print(f"Training [{MODEL_CHOICE.upper()}] on [{DATASET_CHOICE.upper()}] data...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            if MODEL_CHOICE == 'diffusion':
                t = torch.randint(0, TIMESTEPS, (batch_x.shape[0],), device=device).long()
                noise = torch.randn_like(batch_x)
                noisy_x = torch.sqrt(alpha_cumprod[t])[:, None, None] * batch_x + torch.sqrt(1.0 - alpha_cumprod[t])[:, None, None] * noise
                pred_noise = model(noisy_x, t)
                loss = F.mse_loss(pred_noise, noise)
                
            elif MODEL_CHOICE == 'ae' or MODEL_CHOICE == 'unet' or MODEL_CHOICE == 'unet_slim':
                recon = model(batch_x)
                loss = F.mse_loss(recon, batch_x)
                
            elif MODEL_CHOICE == 'vae':
                recon, mu, log_var = model(batch_x)
                recon_loss = F.mse_loss(recon, batch_x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = (recon_loss + kl_loss) / batch_x.size(0)
                    
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss / len(dataloader):.4f}")

    # Step 4: Generation / Reconstruction
    print("\nGenerating/Reconstructing samples for evaluation...")
    if MODEL_CHOICE == 'diffusion':
        generated_samples = sample_diffusion(model, (3, in_channels, SEQ_LEN), device)
    elif MODEL_CHOICE == 'ae' or MODEL_CHOICE == 'unet' or MODEL_CHOICE == 'unet_slim':
        model.eval()
        with torch.no_grad():
            generated_samples = model(raw_data[:3].to(device))
    elif MODEL_CHOICE == 'vae':
        generated_samples = sample_vae(model, 3, in_channels, SEQ_LEN, device)

    # Step 5: Plotting Results
    for show_channel in range(in_channels):
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=True)
        for i in range(3):
            # Real
            axes[0, i].plot(raw_data[i, show_channel].cpu().numpy(), color='tab:blue', lw=2)
            axes[0, i].set_title(f"Real Sample {i+1}")
            axes[0, i].grid(True, alpha=0.3)
            
            # Generated / Reconstructed
            axes[1, i].plot(generated_samples[i, show_channel].cpu().numpy(), color='tab:orange', lw=2)
            axes[1, i].set_title(f"{MODEL_CHOICE.upper()} Output {i+1}")
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(f"Model: {MODEL_CHOICE.upper()} | Dataset: {DATASET_CHOICE.upper()} (Channel {show_channel})", fontsize=14)
        plt.tight_layout()
        plt.show()