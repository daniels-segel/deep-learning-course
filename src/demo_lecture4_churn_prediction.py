import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from contextlib import redirect_stdout

def fix_data(df, channels_used):

    # don't use these columns, don't even try
    df = df.drop(columns=['customerID','InternetService','PaymentMethod','MultipleLines','OnlineSecurity'])
    # turn strings into numbers
    df = df.replace({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 0, 'Month-to-month': 0, 'One year': 1, 'Two year': 2, 'No internet service' : 2})
    # Handles missing space characters:
    df = df.apply(pd.to_numeric, errors='coerce')#.fillna(0))
    # omit lines with missing entries: NaN, None, or NaT (not a time)
    df = df.dropna(subset=channels_used)

    return df[channels_used]

def read_data_basic(data_input_file):

    channels_used = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TechSupport', 'Churn']
    # Load raw tabular data from a CSV file, here treat all channels for completeness
    df_full = pd.read_csv(data_input_file)
    channels_used = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    df = fix_data(df_full, channels_used)

    # Separate features (X) and target label (y)
    X_raw = df.drop(columns=["Churn"]).values
    y_raw = df["Churn"].values

    return df, X_raw, y_raw

def read_data_advanced(data_input_file):

    df_full = pd.read_csv(data_input_file)

    # Drop unique ID column (no predictive value)
    df = df_full.drop(columns=['customerID'])

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
# 2. MODEL DEFINITION & SETUP
# ==========================================
class ChurnMLP(nn.Module):
    def __init__(self, input_dim, hidden_nodes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1),
#            nn.Sigmoid()    # don't need if using BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.net(x)

class ImprovedChurnMLP(nn.Module):
    def __init__(self, input_dim, hidden_nodes, dropout_rate=0.25):
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

def assess_performance(test_predictions, test_probabilities, y_test_tensor, df):

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

def analyze_layers(model, layer_index):
    # W1 shape: (16, 3), W2 shape: (1, 16)
    # model.net[1] is the RELU layer
    W1 = model.net[0].weight.detach().cpu().numpy()
    W2 = model.net[layer_index].weight.detach().numpy().squeeze()

    # Calculate average absolute importance per input feature
    feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
    mean_abs_weights = np.abs(W1).mean(axis=0)

    print("--- Level 1: First-Layer Feature Influence ---")
    for name, importance in zip(feature_names, mean_abs_weights):
        print(f"Feature: {name:<15} | Avg |W1|: {importance:.4f}")

    print("--- Level 2: Hidden Unit Role Decomposition ---")
    for i in range(W2.shape[0]):
        w2_val = W2[i] # Contribution of unit i to output logit
        # Only inspect active neurons that strongly push prediction positive or negative
        if abs(w2_val) > 0.2:
            direction = "Pushes CHURN" if w2_val > 0 else "Pushes RETAIN"
            # weight this hidden neuron has in outcome
            print(f"Neuron {i:02d} ({direction} | W2={w2_val:+.2f}):")
            # weights of 3 inputs in this hidden neuron
            # if tenure is short and bill is high, high risk
            print(f"   <- W1 weights: tenure={W1[i,0]:+.2f}, Monthly={W1[i,1]:+.2f}, Total={W1[i,2]:+.2f}")

def do_plots(model, X_test_tensor, y_test_tensor, pca, plot_name):

    # 1. Create a hook or sub-model to extract 16-D hidden layer activations
    sub_model = nn.Sequential(*list(model.net.children())[:2]) # Keeps Linear(3,16) + ReLU

    model.eval()
    with torch.no_grad():
        hidden_embeddings = sub_model(X_test_tensor).cpu().numpy() # Shape: (N_test, 16)
        y_test_np = y_test_tensor.squeeze().cpu().numpy()

    # 2. Reduce 16D embeddings to 2D using PCA

    embeddings_2d = pca.fit_transform(hidden_embeddings)
    
    hidden_nodes = hidden_embeddings.shape[1]
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
#    plt.show()

    plt.savefig(plot_name, dpi=300, bbox_inches="tight")


def main():

    data_input_file = r'C:\Users\DanielS\.cache\kagglehub\datasets\maqeel732\wa-fn-usec-telco-customer-churn-csv\versions\1\WA_Fn-UseC_-Telco-Customer-Churn.csv'

    # ==========================================
    # 1. READ IN THE DATA
    # ==========================================

    #channels_used = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'Churn']
    hidden_nodes = 16
    epochs = 100
    prediction_threshold = 0.2
    mode_advanced = False
    pca = PCA(n_components=2)
    out_string = 'basic'

    df, X_raw, y_raw = read_data_basic(data_input_file)

    # Train/Test Split
    # stratify=y_raw in train_test_split: Guarantees that both y_train_t and y_test_t preserve the exact same 26.5% churn ratio
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

    # Preprocessing: Scale continuous features (Z-score normalization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # find scale and offset which are saved in class, and also operate on array
    X_test = scaler.transform(X_test)

    # Convert NumPy arrays to PyTorch Tensors (float32 required for linear layers)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    if not mode_advanced:
        layer_index = 2
        model = ChurnMLP(input_dim=X_train.shape[1], hidden_nodes=hidden_nodes)
    else:
        layer_index = 8
        model = ImprovedChurnMLP(input_dim=X_train.shape[1], hidden_nodes=hidden_nodes)
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

    data_output_name = f'output_{out_string}.txt'
    with open(data_output_name, "w") as f:
        with redirect_stdout(f):
            assess_performance(test_predictions, test_probabilities, y_test_tensor, df)
            analyze_layers(model, layer_index)
    plot_name = f'plot_{out_string}.png'
    do_plots(model, X_test_tensor, y_test_tensor, pca, plot_name)

if __name__ == "__main__":

    main()