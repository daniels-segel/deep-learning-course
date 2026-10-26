import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ARCHITECTURE (From previous step)
# ==========================================

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
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Simple1DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        h = self.conv1(F.gelu(x))
        time_emb = self.time_mlp(F.gelu(t))
        h = h + time_emb.unsqueeze(-1)
        h = self.conv2(F.gelu(h))
        return h + self.residual_conv(x)

class TimeSeries1DDenoiser(nn.Module):
    def __init__(self, in_channels=3, base_dim=32, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv1d(in_channels, base_dim, 3, padding=1)
        self.block1 = Simple1DResBlock(base_dim, base_dim * 2, time_emb_dim)
        self.down1 = nn.Conv1d(base_dim * 2, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.bottleneck = Simple1DResBlock(base_dim * 2, base_dim * 4, time_emb_dim)
        self.up1 = nn.ConvTranspose1d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.block2 = Simple1DResBlock(base_dim * 4, base_dim, time_emb_dim)
        self.final_conv = nn.Conv1d(base_dim, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x0 = self.init_conv(x)
        h1 = self.block1(x0, t_emb)
        x_down = self.down1(h1)
        bottleneck = self.bottleneck(x_down, t_emb)
        x_up = self.up1(bottleneck)
        h_cat = torch.cat([x_up, h1], dim=1)
        out = self.block2(h_cat, t_emb)
        return self.final_conv(F.gelu(out))


# ==========================================
# 2. DATASET & SIMULATION ROUTINE
# ==========================================

class EngineeringTimeSeriesDataset(Dataset):
    """Wraps multivariate time series windows into a PyTorch Dataset."""
    def __init__(self, data_tensor):
        # Shape expected: [Num_Samples, Channels, TimeSteps]
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def generate_synthetic_sensor_data(num_samples=500, channels=3, length=64):
    """
    Simulates multi-channel engineering telemetry (e.g., vibration, temperature, pressure).
    Useful for testing code instantly without external files.
    """
    t = torch.linspace(0, 10, length)
    dataset = []
    for _ in range(num_samples):
        # Generate correlated sinusoidal signals with varying frequencies and noise
        freq = torch.rand(1) * 2.0 + 1.0
        phase = torch.rand(1) * math.pi
        
        channel_1 = torch.sin(freq * t + phase) + 0.1 * torch.randn(length) # Vibration
        channel_2 = torch.cos(freq * 0.5 * t) * 2.0 + 0.05 * torch.randn(length) # Temperature
        channel_3 = (t / 10.0) + 0.2 * torch.randn(length) # Pressure degradation trend
        
        sample = torch.stack([channel_1, channel_2, channel_3], dim=0) # Shape: [3, 64]
        dataset.append(sample)
        
    return torch.stack(dataset, dim=0)


# ==========================================
# 3. DIFFUSION SCHEDULES & SAMPLING
# ==========================================

TIMESTEPS = 200
beta = torch.linspace(0.0001, 0.02, TIMESTEPS)
alpha = 1.0 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod = torch.sqrt(alpha_cumprod[t])[:, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alpha_cumprod[t])[:, None, None]
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

@torch.no_grad()
def sample_reverse_diffusion(model, shape, device):
    """Generates brand new time series windows starting from pure Gaussian noise."""
    model.eval()
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    for i in reversed(range(TIMESTEPS)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        # Predict noise using the model
        predicted_noise = model(x, t)
        
        # Step backwards in the Markov chain
        alpha_t = alpha[i]
        alpha_cumprod_t = alpha_cumprod[i]
        beta_t = beta[i]
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        
    return x


# ==========================================
# 4. MAIN EXECUTION ROUTINE
# ==========================================

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Step A: Load or Simulate Data
#     print("Generating synthetic engineering sensor data...")
#     raw_data = generate_synthetic_sensor_data(num_samples=1000, channels=3, length=64)
    
#     # NOTE: If using real C-MAPSS or custom CSV data, load it into a numpy array 
#     # of shape [Num_Samples, Channels, TimeSteps] and pass it to EngineeringTimeSeriesDataset here.

#     dataset = EngineeringTimeSeriesDataset(raw_data)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # Step B: Initialize Model & Optimizer
#     in_channels = 3 # e.g., Vibration, Temp, Pressure
#     model = TimeSeries1DDenoiser(in_channels=in_channels).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     # Step C: Training Loop
#     EPOCHS = 20
#     print(f"Starting training for {EPOCHS} epochs...")
    
#     model.train()
#     for epoch in range(EPOCHS):
#         total_loss = 0.0
#         for batch_x in dataloader:
#             batch_x = batch_x.to(device)
#             optimizer.zero_grad()
            
#             # Sample random timesteps for each sequence in the batch
#             t = torch.randint(0, TIMESTEPS, (batch_x.shape[0],), device=device).long()
            
#             # Forward diffusion (corrupt data)
#             noise = torch.randn_like(batch_x)
#             noisy_x = q_sample(batch_x, t, noise)
            
#             # Predict noise
#             predicted_noise = model(noisy_x, t)
            
#             # Compute loss
#             loss = F.mse_loss(predicted_noise, noise)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

#     # Step D: Generation / Inference
#     print("\nGenerating synthetic engineering time series from latent noise...")
#     sample_shape = (5, in_channels, 64) # Generate 5 synthetic 64-step windows
#     synthetic_samples = sample_reverse_diffusion(model, sample_shape, device)
    
#     print("Generation complete!")
#     print(f"Generated tensor shape: {synthetic_samples.shape}") # Expect [5, 3, 64]

import matplotlib.pyplot as plt

# ==========================================
# 5. VISUALIZATION ROUTINE
# ==========================================

def plot_real_vs_synthetic(real_data, synthetic_data, channel_idx=0):
    """
    Plots 3 timelines from the training set vs 3 generated timelines 
    for a specified sensor channel.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=True)
    
    # Plot Real Data (Row 0)
    for i in range(3):
        axes[0, i].plot(real_data[i, channel_idx].cpu().numpy(), color='tab:blue', lw=2)
        axes[0, i].set_title(f"Real Engineering Sample {i+1}")
        axes[0, i].set_xlabel("Time Steps")
        axes[0, i].set_ylabel("Sensor Amplitude")
        axes[0, i].grid(True, alpha=0.3)
        
    # Plot Generated/Synthetic Data (Row 1)
    for i in range(3):
        axes[1, i].plot(synthetic_data[i, channel_idx].cpu().numpy(), color='tab:orange', lw=2)
        axes[1, i].set_title(f"Diffusion Generated Sample {i+1}")
        axes[1, i].set_xlabel("Time Steps")
        axes[1, i].set_ylabel("Sensor Amplitude")
        axes[1, i].grid(True, alpha=0.3)
        
    plt.suptitle(f"Comparison: Training Set vs. Diffusion Generated (Channel {channel_idx})", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN ROUTINE (Updated with Plotting)
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate Data & Setup Dataloader
    print("Generating synthetic engineering sensor data...")
    raw_data = generate_synthetic_sensor_data(num_samples=1000, channels=3, length=64)
    dataset = EngineeringTimeSeriesDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Initialize Model & Optimizer
    in_channels = 3 
    model = TimeSeries1DDenoiser(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training Loop (Shortened to 5 epochs for demo quickness)
    EPOCHS = 50
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, TIMESTEPS, (batch_x.shape[0],), device=device).long()
            noise = torch.randn_like(batch_x)
            noisy_x = q_sample(batch_x, t, noise)
            predicted_noise = model(noisy_x, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss / len(dataloader):.4f}")

    # 4. Generation / Inference
    print("\nGenerating synthetic engineering time series from latent noise...")
    sample_shape = (3, in_channels, 64) # Generate 3 samples to match plotting layout
    synthetic_samples = sample_reverse_diffusion(model, sample_shape, device)
    
    # 5. Visualize Results
    print("Plotting comparison graph...")
    # Grab the first 3 real samples from the dataset tensor
    real_sample_subset = raw_data[0:3] 
    plot_real_vs_synthetic(real_sample_subset, synthetic_samples, channel_idx=0)