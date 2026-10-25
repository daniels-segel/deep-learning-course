import json
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleGCN(nn.Module):

  def __init__(self, in_features: int, hidden_dim: int, out_dim: int):
    super().__init__()
    # Linear projection layers ($1 \times 1$ convolutions across feature channels)
    self.W1 = nn.Linear(in_features, hidden_dim)
    self.W2 = nn.Linear(hidden_dim, hidden_dim)

    # Final classification head (outputs raw logit for BCEWithLogitsLoss)
    self.classifier = nn.Linear(hidden_dim, out_dim)

  def gcn_layer(
      self, A_hat: torch.Tensor, X: torch.Tensor, linear_layer: nn.Module
  ) -> torch.Tensor:
    """Performs feature projection followed by spatial neighborhood aggregation."""
    h_proj = linear_layer(X)
    return torch.matmul(A_hat, h_proj)

  def forward(self, A_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # 1. Layer 1: 1-hop aggregation + ReLU
    h1 = torch.relu(self.gcn_layer(A_hat, X, self.W1))

    # 2. Layer 2: 2-hop aggregation + ReLU
    h2 = torch.relu(self.gcn_layer(A_hat, h1, self.W2))

    # 3. Global Readout: Sum pooling across all atoms to get a graph-level vector
    graph_vector = torch.sum(h2, dim=0)

    # 4. Classification head (returns raw logit)
    logit = self.classifier(graph_vector)
    return logit


def load_and_normalize_graph(mol_data: dict):
  """Converts JSON node/edge data into PyTorch tensors with symmetric GCN normalization."""
  num_nodes = len(mol_data["nodes"])
  X = torch.tensor(mol_data["nodes"], dtype=torch.float32)

  # Initialize adjacency matrix with self-loops (A + I)
  A = torch.eye(num_nodes)
  for u, v, weight in mol_data["edges"]:
    A[u, v] = weight
    A[v, u] = weight  # Undirected bonds

  # Compute symmetric degree normalization: D^(-1/2) * A * D^(-1/2)
  degrees = torch.sum(A, dim=1)
  d_inv_sqrt = torch.pow(degrees, -0.5)
  d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
  D_inv_sqrt = torch.diag(d_inv_sqrt)

  A_hat = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
  return A_hat, X, mol_data["label"]


if __name__ == "__main__":
  # 1. Load dataset from file
  with open("molecule_input.json", "r") as f:
    dataset = json.load(f)

  # 2. Initialize Model, Loss Function, and Optimizer
  model = SimpleGCN(in_features=2, hidden_dim=16, out_dim=1)
  criterion = nn.BCEWithLogitsLoss()  # Numerically stable binary cross-entropy
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  print("Starting GCN Training Loop:\n" + "-" * 40)

  # 3. Training Loop (e.g., 50 epochs)
  epochs = 50
  for epoch in range(epochs):
    total_loss = 0.0
    model.train()

    for mol_data in dataset:
      optimizer.zero_grad()  # Clear previous gradients

      # Prepare graph tensors
      A_hat, X, true_label = load_and_normalize_graph(mol_data)
      label_tensor = torch.tensor(
          [float(true_label)], dtype=torch.float32
      )

      # Forward pass
      logit = model(A_hat, X)

      # Compute binary cross-entropy loss
      #loss = criterion(logit.unsqueeze(0), label_tensor)
      loss = criterion(logit.squeeze(), label_tensor.squeeze())
      

      # Backward pass (computes gradients via backpropagation)
      loss.backward()

      # Optimizer step (updates weights W1, W2, and classifier)
      optimizer.step()

      total_loss += loss.item()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
      print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f}")

  print("-" * 40 + "\nTraining Complete! Evaluating Final Predictions:\n")

  # 4. Evaluation Pass
  model.eval()
  with torch.no_grad():
    for mol_data in dataset:
      A_hat, X, true_label = load_and_normalize_graph(mol_data)
      logit = model(A_hat, X)
      probability = torch.sigmoid(logit).item()
      prediction = 1 if probability >= 0.5 else 0

      print(
          f"Molecule: {mol_data['id']} | True Label: {true_label} | Pred Prob:"
          f" {probability:.3f} | Pred Class: {prediction}"
      )