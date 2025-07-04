from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import numpy as np
from pyts.classification import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import Counter, defaultdict

def timeseries_classification():
      
    data = np.load("timeseries_data_feats.npz")
    biases = data["biases"]
    targets = data["targets"]
    # losses = data["losses"]
    feats = data["feats"]
    indices_all = data["indices_all"]


    targets_2d = data["targets"].reshape(-1, 1)    # shape (N, 1)


    # Assume feats, targets, biases are numpy arrays with shape:
    # feats: (N, D), targets: (N,), biases: (N,)
    # c = np.concatenate([losses, targets_2d], axis=1)
    # Normalize (standardize) before clustering
    scaler = StandardScaler()
    c = scaler.fit_transform(feats)

    n_clusters = 3  # You can change this
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm="auto", n_init='auto')
    # kmeans = DBSCAN(eps=5.0, min_samples=100)
    cluster_ids = kmeans.fit_predict(feats)

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


    # Labels: 1 if bias-conflicting
    true_labels = (targets != biases).astype(int)

    # Train/test split on losses (timeseries classifier input)
    X_train, X_test, y_train, y_test, feats_train, feats_test, idx_train, idx_test = train_test_split(
        feats, true_labels, feats, indices_all, test_size=0.2, random_state=42, stratify=true_labels
    )

    # Train a time series classifier (e.g., BOSSVS)
    # clf = BOSSVS(window_size=5)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", clf.score(X_test, y_test))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

        # ======= t-SNE colored by (target, bias) group ========
    # Get relevant data for samples predicted as bias-conflicting
    mask_bc_pred = y_pred == 1
    feats_bc = feats#[mask_bc_pred]
    targets_bc = targets#[idx_test]
    biases_bc = biases#[idx_test]

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


timeseries_classification()