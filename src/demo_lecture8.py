import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCN(nn.Module):

  def __init__(self, in_features: int, hidden_dim: int, out_dim: int):
    super().__init__()
    # Linear projection layers acting as 1x1 convolutions across feature channels
    self.W1 = nn.Linear(in_features, hidden_dim)
    self.W2 = nn.Linear(hidden_dim, hidden_dim)

    # Final MLP classification head for graph-level prediction
    self.classifier = nn.Linear(hidden_dim, out_dim)

  def gcn_layer(
      self, A_hat: torch.Tensor, X: torch.Tensor, linear_layer: nn.Module
  ) -> torch.Tensor:
    """Performs feature projection followed by spatial neighborhood aggregation.

    Equation: sigma( A_hat @ (X @ W) )
    """
    # 1. Feature Projection ($1 \times 1$ convolution equivalent)
    h_proj = linear_layer(X)

    # 2. Spatial Aggregation via normalized adjacency matrix multiplication
    return torch.matmul(A_hat, h_proj)

  def forward(self, A_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:

    # Layer 1: Local 1-hop neighbor aggregation + non-linearity
    h1 = F.relu(self.gcn_layer(A_hat, X, self.W1))

    # Layer 2: 2-hop neighbor aggregation + non-linearity
    h2 = F.relu(self.gcn_layer(A_hat, h1, self.W2))

    # Global Readout (Sum Pooling): Collapse variable-sized node matrix into 1 vector
    graph_vector = torch.sum(h2, dim=0)

    # Classification Head: Output prediction probability between 0 and 1
    out = torch.sigmoid(self.classifier(graph_vector))
    return out


def load_and_normalize_graph(mol_data: dict):
  """Converts JSON node/edge data into PyTorch tensors with symmetric GCN normalization."""
  num_nodes = len(mol_data["nodes"])
  X = torch.tensor(mol_data["nodes"], dtype=torch.float32)

  # Initialize adjacency matrix with self-loops (A + I)
  A = torch.eye(num_nodes)
  for u, v, weight in mol_data["edges"]:
    A[u, v] = weight
    A[v, u] = weight  # Undirected chemical bonds

  # Compute symmetric degree normalization: D^(-1/2) * A * D^(-1/2)
  degrees = torch.sum(A, dim=1)
  d_inv_sqrt = torch.pow(degrees, -0.5)
  d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
  D_inv_sqrt = torch.diag(d_inv_sqrt)

  A_hat = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
  return A_hat, X, mol_data["label"]


if __name__ == "__main__":
  # 1. Load data from input file
  with open("molecule_input.json", "r") as f:
    dataset = json.load(f)

  # 2. Instantiate GCN model (2 input atom features -> 16 hidden dimensions -> 1 output class)
  model = SimpleGCN(in_features=2, hidden_dim=16, out_dim=1)
  model.eval()  # Set to evaluation mode for demonstration

  print("Running GCN Forward Pass on Molecular Dataset:\n" + "-" * 50)

  # 3. Process each molecule through the GNN
  for mol_data in dataset:
    A_hat, X, true_label = load_and_normalize_graph(mol_data)

    # Forward pass
    prediction = model(A_hat, X)

    print(f"Molecule ID : {mol_data['id']}")
    print(f"Node Count  : {X.shape[0]} atoms")
    print(f"True Label  : {true_label}")
    print(f"Prediction  : {prediction.item():.4f}\n" + "-" * 50)