import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD & CLEAN DATA (WITH ALL FEATURES)
# ==========================================
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Drop unique ID column (no predictive value)
df = df.drop(columns=['customerID'])

# Convert target 'Churn' to binary 0/1
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Fix blank strings in TotalCharges and drop missing rows
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# One-hot encode categorical columns into 0/1 flags
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target (y)
X_raw = df_encoded.drop(columns=['Churn']).values.astype(np.float32)
y_raw = df_encoded['Churn'].values.astype(np.float32)

# ==========================================
# 2. SPLIT, SCALE & CREATE TENSORS
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the PyTorch tensors explicitly
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ==========================================
# 3. DEFINE IMPROVED ARCHITECTURE
# ==========================================
class ImprovedChurnMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = ImprovedChurnMLP(input_dim=X_train_t.shape[1])

# ==========================================
# 4. LOSS (WITH CLASS WEIGHTING) & OPTIMIZER
# ==========================================
num_neg = (y_train_t == 0).sum()
num_pos = (y_train_t == 1).sum()
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# ==========================================
# 6. PER-CLASS EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t))
    preds = (probs >= 0.5).float().squeeze()
    targets = y_test_t.squeeze()

    churn_mask = (targets == 1.0)
    retain_mask = (targets == 0.0)

    churn_acc = (preds[churn_mask] == 1.0).float().mean().item() * 100
    retain_acc = (preds[retain_mask] == 0.0).float().mean().item() * 100
    overall_acc = (preds == targets).float().mean().item() * 100

print("\n--- Test Results ---")
print(f"Overall Accuracy:    {overall_acc:.2f}%")
print(f"Churn Recall:        {churn_acc:.2f}%")
print(f"Retained Recall:     {retain_acc:.2f}%")