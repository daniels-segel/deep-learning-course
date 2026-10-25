import torch
import torch.nn as nn


class FullyConnectedResBlock(nn.Module):
    """
    A 2-Layer Fully Connected Residual Block:
    f(x) = Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm
    Output = ReLU(x + f(x))
    """
    def __init__(self, dim: int):
        super().__init__()
        # Branch B: 2-layer MLP parameter path
        self.f = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is Branch A (Identity), self.f(x) is Branch B (2-layer transform)
        # Derivative: I + W2 * diag(ReLU') * W1
        return self.relu(x + self.f(x))


class TwoResBlockNetwork(nn.Module):
    """
    Complete network architecture with:
    1. Input Projection Layer
    2. Residual Block 1 (2 FC layers)
    3. Residual Block 2 (2 FC layers)
    4. Output Classification/Regression Head
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # 1. Project raw inputs to hidden dimension space
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. First Residual Block (2 FC layers inside)
        self.block1 = FullyConnectedResBlock(dim=hidden_dim)
        
        # 3. Second Residual Block (2 FC layers inside)
        self.block2 = FullyConnectedResBlock(dim=hidden_dim)
        
        # 4. Final output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.block1(x)  # Residual Block 1
        x = self.block2(x)  # Residual Block 2
        x = self.output_layer(x)
        return x


# --- Example Usage & Verification ---
if __name__ == "__main__":
    batch_size = 32
    input_features = 20
    hidden_dim = 128
    num_classes = 5

    # Initialize model
    model = TwoResBlockNetwork(
        input_dim=input_features, 
        hidden_dim=hidden_dim, 
        output_dim=num_classes
    )

    # Dummy input batch
    dummy_x = torch.randn(batch_size, input_features)
    
    # Forward pass
    output = model(dummy_x)

    print("Model Architecture:")
    print(model)
    print("\n" + "="*50 + "\n")
    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {output.shape}")


import struct

record_format = struct.Struct("<i f 16s")  # int32, float32, 16-byte string (little-endian)

with open("data.bin", "rb") as file:
    while chunk := file.read(record_format.size):
        id_num, score, name = record_format.unpack(chunk)
        print(f"ID: {id_num}, Score: {score:.2f}, Name: {name.decode().rstrip(chr(0))}")