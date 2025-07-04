import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from scipy.stats import mode

# Load data
data = np.load("timeseries_data_feats.npz")
biases = data["biases"]
targets = data["targets"]
losses = data["losses"]

# Normalize features
scaler = MinMaxScaler()
losses = scaler.fit_transform(losses)

# Labels: 1 if bias-conflicting, 0 if bias-aligned
labels = (targets != biases).astype(int)

# Train/test split
X_rest, X_test, targets_rest, targets_test, y_rest, y_test = train_test_split(
    losses, targets, labels, test_size=0.2, random_state=42, stratify=labels
)

# Get indices
bc_indices = np.where(y_rest == 1)[0]
ba_indices = np.where(y_rest == 0)[0]
num_bc = len(bc_indices)
num_ba = len(ba_indices)
num_models = (num_ba + num_bc - 1) // num_bc

print(f"Remaining after test split: {len(X_rest)} samples")
print(f" - Bias-conflicting: {num_bc}")
print(f" - Bias-aligned: {num_ba}")
print(f"Training {num_models} models...")

# Prepare PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reshape for LSTM: (samples, sequence_length=1, input_dim)
X_rest = X_rest[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Store predictions
test_preds = []

# Shuffle BA indices
np.random.seed(42)
np.random.shuffle(ba_indices)

# LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

# Training loop
def train_model(X_train, y_train, input_dim, epochs=20):
    model = SimpleLSTM(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model

for i in range(num_models):
    start_idx = i * num_bc
    end_idx = min(start_idx + num_bc, num_ba)
    ba_chunk = ba_indices[start_idx:end_idx]
    train_indices = np.concatenate([bc_indices, ba_chunk])

    X_train = X_rest[train_indices]
    y_train = y_rest[train_indices]

    model = train_model(X_train, y_train, input_dim=X_train.shape[2])

    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor).cpu().numpy()
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    test_preds.append(y_pred)

# Majority vote
test_preds = np.stack(test_preds, axis=0)
final_preds, _ = mode(test_preds, axis=0)
final_preds = final_preds.squeeze()

# Evaluation
accuracy = (final_preds == y_test).mean()
cm = confusion_matrix(y_test, final_preds, labels=[0, 1])
precision = precision_score(y_test, final_preds, pos_label=1)
recall = recall_score(y_test, final_preds, pos_label=1)

print("\n=== FINAL RESULTS ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision (for bias-conflicting class=1): {precision:.4f}")
print(f"Recall (for bias-conflicting class=1): {recall:.4f}")
print("Confusion Matrix:")
print(cm)
