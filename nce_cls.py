from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
# Load .npz file
data = np.load("nce_ce_outputs.npz")

targets = data["targets"]
biases = data["biases"]
nce_logits = data["nce_logits"]
ce_logits = data["ce_logits"]
nce_preds = data["nce_preds"]
ce_preds = data["ce_preds"]

nce_preds_2d = data["nce_preds"].reshape(-1, 1)  # shape (N, 1)
ce_preds_2d = data["ce_preds"].reshape(-1, 1)    # shape (N, 1)
targets_2d = data["targets"].reshape(-1, 1)    # shape (N, 1)

nce_correct = nce_preds == targets
ce_correct = ce_preds == targets
nce_correct = nce_correct.reshape(-1, 1)  # shape (N, 1)
ce_correct = ce_correct.reshape(-1, 1)    # shape (N, 1)
features = np.concatenate([nce_correct, ce_correct, targets_2d], axis=1)  # shape (N, 2 * num_classes)

# Assume feats, targets, biases are numpy arrays with shape:
# feats: (N, D), targets: (N,), biases: (N,)
# Normalize (standardize) before clustering
scaler = StandardScaler()
c = scaler.fit_transform(features)

n_clusters = 10  # You can change this
kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm="auto", n_init='auto')
# kmeans = DBSCAN(eps=5.0, min_samples=100)
cluster_ids = kmeans.fit_predict(features)

# Create a dictionary to store subgroup counts per cluster
cluster_subgroup_counts = defaultdict(lambda: Counter())

# Build group labels (as strings: '00', '01', etc.)
subgroups = np.array([f"{t}{b}" for t, b in zip(targets, biases)])

# Count each subgroup per cluster
for cluster_id, group in zip(cluster_ids, subgroups):
    cluster_subgroup_counts[cluster_id][group] += 1

# Print results
for cluster_id in sorted(cluster_subgroup_counts):
    print(f"\nCluster {cluster_id}:")
    for group in ['00', '01', '10', '11']:
        count = cluster_subgroup_counts[cluster_id][group]
        print(f"  Group {group}: {count}")


from sklearn.utils import resample

# Create binary labels
labels = (targets == biases).astype(int)  # 1: BA, 0: BC

# Separate BA and BC
ba_indices = np.where(labels == 1)[0]
bc_indices = np.where(labels == 0)[0]

# Undersample BA to match BC count
np.random.seed(43)
ba_sampled = np.random.choice(ba_indices, size=len(bc_indices), replace=False)

# Combine balanced indices
balanced_indices = np.concatenate([bc_indices, ba_sampled])
np.random.shuffle(balanced_indices)




# Get relevant data for samples predicted as bias-conflicting
# mask_bc_pred = y_pred == 1


feats_bc = features#[balanced_indices]
targets_bc = targets#[balanced_indices]
biases_bc = biases#[balanced_indices]
# Optional: subsample for clarity
# max_plot = 1000
# if feats_bc.shape[0] > max_plot:
#     sel = np.random.choice(feats_bc.shape[0], max_plot, replace=False)
#     feats_bc = feats_bc[sel]
#     targets_bc = targets_bc[sel]
#     biases_bc = biases_bc[sel]

# Group labels as strings: '00', '01', etc.
group_labels = np.array([f"{t}{b}" for t, b in zip(targets_bc, biases_bc)])

print(f"Running t-SNE on {len(feats_bc)} predicted bias-conflicting samples...")
tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
tsne_proj = tsne.fit_transform(feats_bc)

# Unique groups and color mapping
unique_groups = sorted(set(group_labels))
colors = ['red', 'blue', 'blue', 'red']
group_color_map = {grp: colors[i] for i, grp in enumerate(unique_groups)}

plt.figure(figsize=(8, 6))
for grp in unique_groups:
    idxs = np.where(group_labels == grp)[0]
    plt.scatter(
        tsne_proj[idxs, 0], tsne_proj[idxs, 1],
        label=f'Group {grp}',
        color=group_color_map[grp],
        s=10, alpha=0.4
    )

plt.title("t-SNE of Predicted Bias-Conflicting Samples\nColored by (target, bias) Group")
plt.xlabel("t-SNE-1")
plt.ylabel("t-SNE-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_bc_by_group.png")
plt.show()
print("Saved t-SNE plot as 'tsne_bc_by_group.png'")
