import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool

class SparseGCNLayer(nn.Module):

  def __init__(self, in_features, out_features):
    super().__init__()
    # Our shared feature transformation weight matrix W
    self.linear = nn.Linear(in_features, out_features, bias=False)

  def forward(self, x, edge_index):
    # 1. Feature projection: XW (dense matrix multiplication)
    h = self.linear(x)

    # 2. Add self-loops to the edge index so nodes aggregate their own features
    num_nodes = x.size(0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 3. Compute symmetric normalization coefficients: D^{-1/2} * A * D^{-1/2}
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 4. Construct a sparse adjacency tensor
    adj_sparse = torch.sparse_coo_tensor(
        edge_index, norm, size=(num_nodes, num_nodes)
    )

    # 5. Sparse aggregation: multiply sparse adjacency by projected features
    out = torch.sparse.mm(adj_sparse, h)
    return out

class PyG_CustomGCN(nn.Module):

  def __init__(self, in_features, hidden_dim, out_dim):
    super().__init__()
    self.conv1 = SparseGCNLayer(in_features, hidden_dim)
    self.conv2 = SparseGCNLayer(hidden_dim, hidden_dim)
    #self.classifier = nn.Linear(hidden_dim, out_dim)
    self.classifier = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, out_dim))
 
  def forward(self, x, edge_index, batch):
    # Layer 1 + non-linearity
    x = self.conv1(x, edge_index)
    x = torch.relu(x)

    # Layer 2 + non-linearity
    x = self.conv2(x, edge_index)
    x = torch.relu(x)

    # Global Readout: Collapse node-level vectors into a graph-level vector
    # 'batch' tells the pool function which nodes belong to which graph in the batch
    x = global_mean_pool(x, batch)

    # Final classification head
    out = self.classifier(x)
    return out


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

  dataset = TUDataset(root='./data/PROTEINS', name='PROTEINS')

  print(f'Number of graphs: {len(dataset)}')
  print(f'Node feature dimension: {dataset.num_features}')
  print(f'Number of classes: {dataset.num_classes}')

  train_size = int(0.8 * len(dataset))

  train_dataset = dataset[:train_size]
  val_dataset = dataset[train_size:]

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  # 2. Initialize Model, Loss Function, and Optimizer
  model = PyG_CustomGCN(
    in_features=dataset.num_features, hidden_dim=32, out_dim=1)
  criterion = nn.BCEWithLogitsLoss()  # Numerically stable binary cross-entropy
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  print("Starting GCN Training Loop:\n" + "-" * 40)

  # 3. Training Loop (e.g., 50 epochs)
  epochs = 100
  for epoch in range(epochs):
    total_train_loss = 0.0
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()

        # PyG's DataLoader automatically bundles batched graphs together:
        # batch.x = all node features combined
        # batch.edge_index = all edges combined (with shifted indices)
        # batch.batch = tensor mapping each node to its source graph ID
        # batch.y = graph-level labels

        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits.squeeze(), batch.y.float())

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(
      f'Epoch {epoch+1}/{epochs} | Loss:'
      f' {total_train_loss / len(train_loader):.4f}')


  print("-" * 40 + "\nTraining Complete! Evaluating Final Predictions:\n")

  # 4. Evaluation Pass
  model.eval()  # Disables dropout, freezes batch norm stats
  total_val_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():  # Disables gradient computation to save memory/speed up
    for batch in val_loader:
      logits = model(batch.x, batch.edge_index, batch.batch)
      target = batch.y.float()

      # Compute validation loss
      val_loss = criterion(logits.squeeze(), target)
      total_val_loss += val_loss.item()

      # Compute binary classification accuracy (threshold at 0.5 probability)
      probs = torch.sigmoid(logits.squeeze())
      preds = (probs >= 0.5).float()

      # Handle edge case where a batch size might be 1 (unsqueeze safety)
      if preds.ndim == 0:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

      correct += (preds == target).sum().item()
      total += target.size(0)

  avg_val_loss = total_val_loss / len(val_loader)
  val_accuracy = (correct / total) * 100.0 if total > 0 else 0.0

  print(
      f'Epoch {epoch+1:02d}/{epochs} | '
      f'Train Loss: {avg_train_loss:.4f} | '
      f'Val Loss: {avg_val_loss:.4f} | '
      f'Val Accuracy: {val_accuracy:.2f}%'
  )