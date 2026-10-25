import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

n_epochs = 50
timesteps = 140
num_samples = 1000

# demo of use of NN for data analysis

# force signal into bottleneck of 8 numbers then reconstruct

# 1. Generate Synthetic Heartbeat-like Signal Data Locally (Zero Network Dependency)
def generate_synthetic_ecg(num_samples, timesteps):
    t = np.linspace(0, 2 * np.pi, timesteps)
    
    # Normal signals: Consistent combination of sine waves + small P-Q-R-S-T simulated pulses
    base_signal = np.sin(t) + 0.5 * np.cos(2 * t) + 2.0 * np.exp(-((t - 3)**2) / 0.1)
    
    # Generate normal dataset with minor Gaussian noise
    normal_data = base_signal + np.random.normal(0, 0.1, (num_samples, timesteps))
    
    # Generate anomaly dataset: Introduce unexpected spikes and altered frequencies
    anomaly_data = base_signal + np.random.normal(0, 0.1, (200, timesteps))
    for i in range(200):
        # Add random sharp arrhythmia spike or inverted pulse
        spike_idx = np.random.randint(20, 120)
        anomaly_data[i, spike_idx:spike_idx+15] += np.random.choice([-3.0, 3.0])
        
    return normal_data.astype(np.float32), anomaly_data.astype(np.float32)

print("Generating synthetic time-series signals...")
X_normal, X_anomaly = generate_synthetic_ecg(num_samples=1000, timesteps=140)

# Train ONLY on Normal signals
X_train, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)
X_test_anomaly = X_anomaly

# Convert to PyTorch Tensors
x_train = torch.tensor(X_train)
x_test_normal = torch.tensor(X_test_normal)
x_test_anomaly = torch.tensor(X_test_anomaly)

# 2. Define Autoencoder Architecture
class SignalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__() # sets up all data structures, settings, hooks
        self.encoder = nn.Sequential(
            nn.Linear(timesteps, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # Bottleneck representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, timesteps)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = SignalAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Train Model Live (20 Epochs)
print("Training Autoencoder on Normal Signals...")
for epoch in range(1, n_epochs):
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, x_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/20] - Loss: {loss.item():.4f}")

# 4. Generate Visual Comparison for the Class
model.eval()
with torch.no_grad():
    recon_normal = model(x_test_normal[0:1]).numpy()
    recon_anomaly = model(x_test_anomaly[0:1]).numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Normal Signal
axes[0].plot(x_test_normal[0].numpy(), label="Original Normal", color="blue")
axes[0].plot(recon_normal[0], label="Reconstruction", color="orange", linestyle="--")
axes[0].set_title("Normal Signal (Low Reconstruction Error)")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Anomaly Signal
axes[1].plot(x_test_anomaly[0].numpy(), label="Original Anomaly", color="red")
axes[1].plot(recon_anomaly[0], label="Reconstruction", color="black", linestyle="--")
axes[1].set_title("Abnormal Signal (High Reconstruction Error)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f'normal reconstruction accuracy = {np.mean(x_test_normal[0].numpy()-recon_normal[0])}')
print(f'abnormal reconstruction accuracy = {np.mean(x_test_anomaly[0].numpy()-recon_anomaly[0])}')