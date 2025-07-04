import numpy as np
from pyts.classification import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from pyts.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score

# Load data
data = np.load("timeseries_data_feats.npz")
biases = data["biases"]
targets = data["targets"]
losses = data["losses"]

# Create labels
labels = (targets != biases).astype(int)

# First, split 20% as test set (no balancing)
X_rest, X_test, targets_rest, targets_test, y_rest, y_test = train_test_split(
    losses, targets, labels, test_size=0.2, random_state=42, stratify=labels
)

# Compute class means on X_rest
target0_mean = X_rest[targets_rest == 0].mean(axis=0)
target1_mean = X_rest[targets_rest == 1].mean(axis=0)

# Center X_rest
X_rest_centered = X_rest.copy()
X_rest_centered[targets_rest == 0] -= target0_mean
X_rest_centered[targets_rest == 1] -= target1_mean
X_rest = X_rest_centered.copy()

# Center X_test using the SAME class means (from training data only)
X_test_centered = X_test.copy()
X_test_centered[targets_test == 0] -= target0_mean
X_test_centered[targets_test == 1] -= target1_mean
X_test = X_test_centered.copy()

# Get indices for BC (bias-conflicting) and BA (bias-aligned) from remaining 80%
bc_indices = np.where(y_rest == 1)[0]
ba_indices = np.where(y_rest == 0)[0]

num_bc = len(bc_indices)
num_ba = len(ba_indices)

print(f"Remaining after test split: {len(X_rest)} samples")
print(f" - Bias-conflicting: {num_bc}")
print(f" - Bias-aligned: {num_ba}")

# Split BA samples into approximately equal parts to match number of BC samples
num_models = (num_ba + num_bc - 1) // num_bc  # round up
print(f"Training {num_models} models...")

# Normalize
# scaler = StandardScaler()
# X_rest = scaler.fit_transform(X_rest)
# X_test = scaler.transform(X_test)

# Store predictions from all models
test_preds = []

# Shuffle BA indices to randomize chunks
np.random.seed(42)
np.random.shuffle(ba_indices)

for i in range(num_models):
    # Select a slice of BA samples
    start_idx = i * num_bc
    end_idx = min(start_idx + num_bc, num_ba)
    ba_chunk = ba_indices[start_idx:end_idx]

    # Combine this BA chunk with all BC samples
    train_indices = np.concatenate([bc_indices, ba_chunk])

    # Prepare data
    X_train = X_rest[train_indices]
    y_train = y_rest[train_indices]

    # Initialize and train model
    clf = KNeighborsClassifier()
    # clf = BOSSVS(window_size=28)
    # clf = SAXVSM(window_size=16, word_size=12, n_bins=5, strategy='normal')
    # clf = LearningShapelets(random_state=42, tol=0.001)
    # clf =  TimeSeriesForest(random_state=43)
    # clf = TSBF(random_state=43)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)
    test_preds.append(y_pred)

# Stack all predictions: shape (num_models, num_samples)
test_preds = np.stack(test_preds, axis=0)

# Majority vote across models
final_preds, _ = mode(test_preds, axis=0)
final_preds = final_preds.squeeze()

# Evaluate
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