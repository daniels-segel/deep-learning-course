import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def fix_data(df, channels_used):

    df = df.drop(columns=['customerID','InternetService','PaymentMethod','MultipleLines','OnlineSecurity'])
    df = df.replace({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 0, 'Month-to-month': 0, 'One year': 1, 'Two year': 2, 'No internet service' : 2})
    df = df.apply(pd.to_numeric, errors='coerce')#.fillna(0))
    df = df.dropna(subset=channels_used)

    return df[channels_used]


# ==========================================
# 1. READ IN THE DATA
# ==========================================

#channels_used = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'Churn']
channels_used = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TechSupport', 'Churn']
hidden_nodes = 16
epochs = 100
prediction_threshold = 0.2

pca = PCA(n_components=2)
# Load raw tabular data from a CSV file, here treat all channels for completeness
df_full = pd.read_csv(r"WA_Fn-UseC_-Telco-Customer-Churn.csv")
channels_used = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
df = fix_data(df_full, channels_used)

# Separate features (X) and target label (y)
X_raw = df.drop(columns=["Churn"]).values
y_raw = df["Churn"].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Preprocessing: Scale continuous features (Z-score normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # find scale and offset which are saved in class, and also operate on array
X_test = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch Tensors (float32 required for linear layers)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# ==========================================
# 2. MODEL DEFINITION & SETUP
# ==========================================
class ChurnMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1),
#            nn.Sigmoid()    # don't need if using BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.net(x)

model = ChurnMLP(input_dim=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ==========================================
# 3. TRAINING LOOP
# ==========================================
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    model.train()
    
    # Forward Pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # Backward Pass & Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ==========================================
# 4. SEE THE RESULTS (EVALUATION / INFERENCE)
# ==========================================
model.eval() # Set model to evaluation mode

with torch.no_grad(): # Disable gradient computation for speed & memory efficiency
    # Get continuous probability outputs (between 0.0 and 1.0)
    test_probabilities = model(X_test_tensor)
    
    # Convert probabilities to discrete binary predictions (0 or 1) using a 0.5 threshold
    test_predictions = (test_probabilities >= prediction_threshold).float()
    
    # Calculate classification accuracy
    correct = (test_predictions == y_test_tensor).sum().item()
    accuracy = correct / len(y_test_tensor)

    preds = test_predictions.squeeze()
    targets = y_test_tensor.squeeze()

    # Create boolean masks for each class
    churn_mask = (targets == 1.0)
    not_churn_mask = (targets == 0.0)

    # 1. Churn Accuracy (Recall for Class 1)
    churn_correct = (preds[churn_mask] == 1.0).sum().item()
    churn_total = churn_mask.sum().item()
    churn_accuracy = churn_correct / churn_total if churn_total > 0 else 0.0

    # 2. Not-Churn Accuracy (Recall for Class 0 / Specificity)
    not_churn_correct = (preds[not_churn_mask] == 0.0).sum().item()
    not_churn_total = not_churn_mask.sum().item()
    not_churn_accuracy = not_churn_correct / not_churn_total if not_churn_total > 0 else 0.0

    # 3. Overall Accuracy
    overall_accuracy = (preds == targets).sum().item() / len(targets)

# Print Results
print("--- Class-wise Accuracy Breakdown ---")
churn_percentage = df['Churn'].mean() * 100

print(f"Total Dataset: {len(df)} rows")
print(f"Churned ('Yes'): {churn_percentage:.2f}%")
print(f"Overall Accuracy:       {overall_accuracy * 100:.2f}%")
print(f"Churn ('Yes') Accuracy:    {churn_accuracy * 100:.2f}% ({churn_correct}/{churn_total} caught)")
print(f"Not Churn ('No') Accuracy: {not_churn_accuracy * 100:.2f}% ({not_churn_correct}/{not_churn_total} caught)")

# Print direct sample predictions
print("--- Sample Predictions ---")
print("Raw Probabilities (First 5):", test_probabilities[:5].squeeze().tolist())
print("Predicted Classes (First 5):", test_predictions[:5].squeeze().tolist())
print("Actual Ground Truth (First 5):", y_test_tensor[:5].squeeze().tolist())
print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

# plt.plot((test_predictions*1.1).squeeze().tolist(),'o-')
# plt.plot(y_test_tensor.squeeze().tolist(),'ro-')
# plt.legend(['test','GT'])
# plt.show()

# W1 shape: (16, 3), W2 shape: (1, 16)
# model.net[1] is the RELU layer
W1 = model.net[0].weight.detach().cpu().numpy()
W2 = model.net[2].weight.detach().numpy().squeeze()

# Calculate average absolute importance per input feature
feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
mean_abs_weights = np.abs(W1).mean(axis=0)

print("--- Level 1: First-Layer Feature Influence ---")
for name, importance in zip(feature_names, mean_abs_weights):
    print(f"Feature: {name:<15} | Avg |W1|: {importance:.4f}")

print("--- Level 2: Hidden Unit Role Decomposition ---")
for i in range(hidden_nodes):
    w2_val = W2[i] # Contribution of unit i to output logit
    # Only inspect active neurons that strongly push prediction positive or negative
    if abs(w2_val) > 0.2:
        direction = "Pushes CHURN" if w2_val > 0 else "Pushes RETAIN"
        # weight this hidden neuron has in outcome
        print(f"Neuron {i:02d} ({direction} | W2={w2_val:+.2f}):")
        # weights of 3 inputs in this hidden neuron
        # if tenure is short and bill is high, high risk
        print(f"   <- W1 weights: tenure={W1[i,0]:+.2f}, Monthly={W1[i,1]:+.2f}, Total={W1[i,2]:+.2f}")

# 1. Create a hook or sub-model to extract 16-D hidden layer activations
sub_model = nn.Sequential(*list(model.net.children())[:2]) # Keeps Linear(3,16) + ReLU

model.eval()
with torch.no_grad():
    hidden_embeddings = sub_model(X_test_tensor).cpu().numpy() # Shape: (N_test, 16)
    y_test_np = y_test_tensor.squeeze().cpu().numpy()

# 2. Reduce 16D embeddings to 2D using PCA

embeddings_2d = pca.fit_transform(hidden_embeddings)

# 3. Plot the learned latent space
plt.figure(figsize=(8, 6))
plt.scatter(
    embeddings_2d[y_test_np == 0, 0], 
    embeddings_2d[y_test_np == 0, 1], 
    c='blue', alpha=0.4, label='Retained (0)', s=20
)
plt.scatter(
    embeddings_2d[y_test_np == 1, 0], 
    embeddings_2d[y_test_np == 1, 1], 
    c='red', alpha=0.6, label='Churned (1)', s=20
)
plt.title(f"Learned Hidden Layer Representations ({hidden_nodes}D -> 2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()