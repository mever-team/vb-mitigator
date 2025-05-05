from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore
from itertools import combinations
from tqdm import tqdm

# ====== CONFIG ======
DATA_PATH = "timeseries_data_feats.npz"
SAVE_PATH = "low_corr_pairs_indices.npy"
CORR_THRESHOLD = -0.50
# =====================

# Load data
print("Loading data...")
data = np.load(DATA_PATH)
biases = data["biases"]       # (N,)
targets = data["targets"]     # (N,)
losses = data["losses"]       # (N, T)
indices_all = data["indices_all"]

# Normalize time series for correlation computation
print("Z-scoring losses...")
losses = zscore(losses, axis=1)

# Compute full correlation matrix (N x N)
print("Computing full correlation matrix...")
corr_matrix = np.corrcoef(losses)

mean_corr = np.mean(corr_matrix,axis=1)

mean_corr = np.array(mean_corr)
mean_corr =  (mean_corr - mean_corr.min()) / (mean_corr.max() - mean_corr.min())
inv_corr = np.log(1 * mean_corr + 1e-9)
# inv_corr = (mean_corr - mean_corr.min()) / (mean_corr.max() - mean_corr.min())

# Step 2: Map weights to sample indices
weights_dict = {int(idx): float(weight) for idx, weight in zip(indices_all, inv_corr)}

# Step 3: Save as .npy or .pkl file
np.save("sample_weights_dict.npy", weights_dict)

# Compute boolean masks
bias_aligned_mask = (targets == biases)
bias_conflicting_mask = ~bias_aligned_mask  # opposite

# Split mean correlations
mean_corr_ba = mean_corr[bias_aligned_mask]
mean_corr_bc = mean_corr[bias_conflicting_mask]

# Plot histogram
plt.figure(figsize=(10, 6))
bins = np.linspace(min(mean_corr), max(mean_corr), 50)

plt.hist(mean_corr_ba, bins=bins, alpha=0.6, label='Bias-Aligned (target == bias)')
plt.hist(mean_corr_bc, bins=bins, alpha=0.6, label='Bias-Conflicting (target != bias)')

plt.xlabel("Mean Correlation with Other Samples")
plt.ylabel("Number of Samples")
plt.title("Histogram of Mean Correlation by Bias Alignment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_correlation_hist_by_bias_alignment.png")
plt.close()

# Get upper triangle indices (excluding diagonal)
n = len(losses)
i_indices, j_indices = np.triu_indices(n, k=1)

# Filter low-correlation pairs
print("Filtering low-correlation pairs...")
low_corr_mask = corr_matrix[i_indices, j_indices] < CORR_THRESHOLD
i_low = i_indices[low_corr_mask]
j_low = j_indices[low_corr_mask]
low_corr_pairs = np.stack((indices_all[i_low], indices_all[j_low]), axis=1)

# Save to file
np.save(SAVE_PATH, low_corr_pairs)
print(f"Saved {len(low_corr_pairs)} low-correlation pairs to {SAVE_PATH}")

# ====== Evaluation ======
print("Evaluating pair properties...")
b1, b2 = biases[i_low], biases[j_low]
t1, t2 = targets[i_low], targets[j_low]

same_target_diff_bias = np.sum((t1 == t2) & (b1 != b2))
same_bias_diff_target = np.sum((b1 == b2) & (t1 != t2))
diff_both = np.sum((t1 != t2) & (b1 != b2))
same_both = np.sum((t1 == t2) & (b1 == b2))

total = len(low_corr_pairs)
print("\n=== Performance Evaluation ===")
print(f"Total low-corr pairs: {total}")
print(f"Same target, diff bias: {same_target_diff_bias} ({same_target_diff_bias/total:.2%})")
print(f"Diff target, same bias: {same_bias_diff_target} ({same_bias_diff_target/total:.2%})")
print(f"Diff target, diff bias: {diff_both} ({diff_both/total:.2%})")
print(f"Same target, same bias: {same_both} ({same_both/total:.2%})")

