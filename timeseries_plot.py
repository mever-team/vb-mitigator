import numpy as np
import matplotlib.pyplot as plt

# Load your data
data = np.load("timeseries_data.npz")
biases = data["biases"]
targets = data["targets"]
losses = data["losses"]  # (N_samples, T_timepoints)


# Compute class means on X_rest
target0_mean = losses[targets == 0].mean(axis=0)
target1_mean = losses[targets == 1].mean(axis=0)

# Center X_rest
losses_centered = losses.copy()
losses_centered[targets == 0] -= target0_mean
losses_centered[targets == 1] -= target1_mean
losses = losses_centered.copy()


# Group identifiers
group_labels = ['00', '01', '10', '11']

# Find group indices
group_indices = {
    '00': np.where((targets == 0) & (biases == 0))[0],
    '01': np.where((targets == 0) & (biases == 1))[0],
    '10': np.where((targets == 1) & (biases == 0))[0],
    '11': np.where((targets == 1) & (biases == 1))[0],
}

# Compute mean loss curves
group_means = {}
for label, idxs in group_indices.items():
    if len(idxs) > 0:
        group_means[label] = losses[idxs].mean(axis=0)
    else:
        group_means[label] = None  # Handle empty groups

# Plot
plt.figure(figsize=(10, 6))
for label, mean_loss in group_means.items():
    if mean_loss is not None:
        plt.plot(mean_loss, label=f'Group {label}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Loss Curves per (Target, Bias) Group')
plt.legend()
plt.grid(True)

# Save figure
plt.savefig('group_mean_losses.png')
print("Saved figure as 'group_mean_losses.png'")
plt.show()
