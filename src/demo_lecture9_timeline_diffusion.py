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

    def forward(self, time):   # time is a 1D vector the length of the batch
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

def generate_synthetic_sensor_data(num_samples=500, channels=3, length=64):
    """
    Simulates multi-channel engineering telemetry (e.g., vibration, temperature, pressure).
    Useful for testing code instantly without external files.
    """
    t = torch.linspace(0, 10, length)
    dataset = []
    for _ in range(num_samples):
        # Generate correlated sinusoidal signals with varying frequencies and noise
        freq = torch.rand(1) * 4.0 + 1.0
        phase = torch.rand(1) * math.pi

        # acceleration from vibration:
        channel_1 = torch.sin(freq * t + phase) + 0.1 * torch.randn(length) # Vibration
        # temperature:
        channel_2 = torch.cos(freq * 0.5 * t) * 2.0 + 0.05 * torch.randn(length) # Temperature
        # Degradation Index:
        channel_3 = (t / 10.0) + 0.2 * torch.randn(length) # Pressure degradation trend
        
        sample = torch.stack([channel_1, channel_2, channel_3], dim=0) # Shape: [3, 64]
        dataset.append(sample)
        
    return torch.stack(dataset, dim=0)


# ==========================================
# 3. DIFFUSION SCHEDULES & SAMPLING
# ==========================================

TIMESTEPS = 400
# calculate diffusion schedule coefficients:
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

    # change from train to eval mode
    model.eval()
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    for i in reversed(range(TIMESTEPS)):
        # put the same t in for all samples
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


import matplotlib.pyplot as plt

# ==========================================
# 5. VISUALIZATION ROUTINE
# ==========================================

def plot_real_vs_synthetic(real_data, synthetic_data, channel_idx=0):
    """
    Plots 3 timelines from the training set vs 3 generated timelines 
    for a specified sensor channel.
    """
    n_timelines = real_data.shape[0]
    fig, axes = plt.subplots(2, n_timelines, figsize=(15, 6), sharey=True)
    
    # Plot Real Data (Row 0)
    for i in range(n_timelines):
        axes[0, i].plot(real_data[i, channel_idx].cpu().numpy(), color='tab:blue', lw=2)
        axes[0, i].set_title(f"Real Engineering Sample {i+1}")
        axes[0, i].set_xlabel("Time Steps")
        axes[0, i].set_ylabel("Sensor Amplitude")
        axes[0, i].grid(True, alpha=0.3)
        
    # Plot Generated/Synthetic Data (Row 1)
    for i in range(n_timelines):
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

    SAMPLE_LENGTH = 64
    # 1. Generate Data & Setup Dataloader
    print("Generating synthetic engineering sensor data...")
    raw_data = generate_synthetic_sensor_data(num_samples=1000, channels=3, length=SAMPLE_LENGTH)

    # Normalize each channel independently to [-1, 1]
    min_val = raw_data.amin(dim=-1, keepdim=True)
    max_val = amax = raw_data.amax(dim=-1, keepdim=True)
    raw_data = 2.0 * (raw_data - min_val) / (max_val - min_val + 1e-5) - 1.0
    
    dataloader = DataLoader(raw_data, batch_size=32, shuffle=True)

    # 2. Initialize Model & Optimizer
    in_channels = 3 
    model = TimeSeries1DDenoiser(in_channels=in_channels, base_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training Loop (Shortened to 5 epochs for demo quickness)
    EPOCHS = 100
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            # randomly assign a different diffusion timestep to each individual sequence in the batch:
            t = torch.randint(0, TIMESTEPS, (batch_x.shape[0],), device=device).long()
            noise = torch.randn_like(batch_x)
            # add noise to sample according to it's assigned timestep:
            noisy_x = q_sample(batch_x, t, noise)
            # model predicts how much noise this sample contains:
            predicted_noise = model(noisy_x, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss / len(dataloader):.4f}")

    # 4. Generation / Inference
    N_SAMPLES = 5
    show_channel = 1

    print("\nGenerating synthetic engineering time series from latent noise...")
    sample_shape = (N_SAMPLES, in_channels, 64) # Generate 3 samples to match plotting layout
    synthetic_samples = sample_reverse_diffusion(model, sample_shape, device)
    
    # 5. Visualize Results
    print("Plotting comparison graph...")
    # Grab the first N_SAMPLES real samples from the dataset tensor
    real_sample_subset = raw_data[0:N_SAMPLES] 
    for show_channel in range(in_channels):
        plot_real_vs_synthetic(real_sample_subset, synthetic_samples, channel_idx=show_channel)