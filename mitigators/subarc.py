import math
import os
from tqdm import tqdm
import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc
import torch.nn.functional as F
from scipy.stats import linregress

import torch
import torch.nn.functional as F
from tools.metrics.utils import AverageMeter
from sklearn.cluster import DBSCAN
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
import matplotlib.patches as mpatches
from collections import defaultdict
from .losses import SubcenterArcMarginProduct, LabelSmoothSoftmaxCEV1, ArcMarginProduct, WBLoss
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

def visualize_clusters(cluster_means, class_ids, title="Cluster Prototypes"):
    """
    Visualize class cluster centers in 2D using t-SNE.
    Args:
        cluster_means: [num_clusters, feat_dim] tensor
        class_ids:     [num_clusters] tensor
    """
    cluster_means_np = cluster_means.numpy()
    class_ids_np = class_ids.numpy()

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(cluster_means_np)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=class_ids_np, cmap='tab20', alpha=0.8)
    plt.colorbar(scatter, label="Class ID")
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("tmp.png")
    plt.show()
def classify_by_k_nearest_clusters(test_feats, cluster_means, class_ids, k=3):
    """
    Classifies test features by majority vote over k nearest cluster centroids.
    
    Args:
        test_feats:     [N, D] tensor of test features
        cluster_means:  [C*K, D] tensor of cluster centers
        class_ids:      [C*K] tensor of class ids corresponding to each cluster
        k:              Number of nearest clusters to consider
    Returns:
        preds: [N] predicted class labels
    """
    preds = []

    for feat in test_feats:
        # Compute L2 distances to all cluster centers
        dists = torch.norm(cluster_means - feat.unsqueeze(0), dim=1)

        # Get indices of k closest clusters
        knn_indices = torch.topk(dists, k=k, largest=False).indices

        # Get class IDs of those clusters
        knn_labels = class_ids[knn_indices]

        # Majority vote
        pred = torch.mode(knn_labels).values
        preds.append(pred)

    return torch.stack(preds)


class LossCurveDataset(Dataset):
    def __init__(self, losses, targets, biases):
        self.losses = torch.tensor(losses, dtype=torch.float32)  # shape (N, T)
        # self.labels = (targets != biases).astype(int)             # bias-conflicting = 1
        self.labels = torch.tensor(targets, dtype=torch.long)
        print(torch.max(self.labels))

    def __len__(self):
        return len(self.losses)

    def __getitem__(self, idx):
        return self.losses[idx].unsqueeze(-1), self.labels[idx]  # (T, 1), label



class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B, hidden)
        out = self.fc(h_n[-1])      # (B, 2)
        return out
    
def train_lstm_classifier(losses, targets, biases, num_epochs=1000):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Prepare dataset
    X_train, X_test, y_train, y_test = train_test_split(losses, targets != biases, test_size=0.2, random_state=42, stratify=targets != biases)
    train_dataset = LossCurveDataset(X_train, y_train, y_train)
    test_dataset = LossCurveDataset(X_test, y_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Model setup
    model = LSTMClassifier(hidden_dim=64).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            preds = model(x)
            # print(torch.max(y))
            # print(torch.max(preds))
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        if epoch%100 == 0:
            # Evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.cuda()
                    preds = model(x).argmax(dim=1).cpu()
                    all_preds.append(preds)
                    all_labels.append(y)

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            print(classification_report(all_labels, all_preds, target_names=["BA", "BC"]))

def compute_classwise_clusters(features, labels, n_classes=100, k=10):
    cluster_means = []
    class_ids = []
    for cls in range(n_classes):
        mask = (labels == cls)
        cls_feats = features[mask]
        if cls_feats.size(0) < k:
            continue  # skip classes with too few samples
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(cls_feats.numpy())
        cluster_means.append(torch.tensor(kmeans.cluster_centers_))
        class_ids += [cls] * k
    cluster_means = torch.cat(cluster_means, dim=0)
    class_ids = torch.tensor(class_ids)
    return cluster_means, class_ids


def collect_features(dataloader, model, device):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["inputs"].to(device)
            labels = batch["targets"]
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                _, features = outputs
            else:
                raise ValueError("Model must return (logits, features)")
            all_feats.append(features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_feats), torch.cat(all_labels)

class SubArcTrainer(BaseTrainer):
    def _setup_resume(self):
        return
    
    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.metric_loss = SubcenterArcMarginProduct(512,4,2,m=0.3)
        # self.metric_loss_default = ArcMarginProduct(512,2,m=0.,m_b=0.)
        # self.metric_loss = ArcMarginProduct(512,2,m=0.3,m_b=0.3,s=100)

        self.wbloss = WBLoss()
        # self.metric_loss = ArcMarginProduct(512,2,m=-0.1,m_b=0.2,s=20) #   0.8780                     0.8910
        # self.metric_loss_default.to(self.device)
        self.metric_loss.to(self.device)


    def _train_iter(self, batch):
        self.optimizer.zero_grad()
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)
        indices = batch["index"]
        
        outputs = self.model(inputs)
        if not isinstance(outputs, tuple):
            raise ValueError("Model output must be a tuple (logits, features)")
        outputs, features = outputs
        # if len(self.preproc_losses.keys()) > 0:

        #     new_margins = (torch.tensor([self.preproc_losses[idx.item()] for idx in indices]))
        #     new_margins = new_margins.to(outputs.dtype) 
        #     loss = self.wbloss(outputs,targets,new_margins.to(self.device))
        #     # logits = self.metric_loss(features,targets,biases, new_margins.to(self.device))
        # else:
        #     loss = self.criterion(outputs, targets)
        #     # logits = self.metric_loss.forward_def(features,targets,biases)
        logits = self.metric_loss(features,targets)
        # logits = outputs
        loss = self.criterion(logits, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        
        return {"train_cls_loss": loss}

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, features = outputs
        logits = self.metric_loss.predict(features,targets)
        # logits = outputs
        loss = self.criterion(logits, targets)
        batch_dict["predictions"] = torch.argmax(logits, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss
    

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None

        for batch in tqdm(self.dataloaders["train"]):
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}

        # # self.losses = {}
        # self.collect_losses_preproc()
        # self.compute_mean_and_normalize()
        return avg_loss
    


    def visualize_losses(self):
        biases = []
        targets = []
        losses = []

        # Collect losses, targets, and bias attributes
        for batch in tqdm(self.dataloaders["train"]):
            bsz = batch["targets"].shape[0]
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]
            batch_losses = torch.tensor([self.losses[idx.item()] for idx in indices])
            # print(batch_losses.shape)
            losses.append(batch_losses)

        # Concatenate all collected data
        biases = torch.cat(biases).numpy()
        targets = torch.cat(targets).numpy()
        losses = torch.cat(losses).numpy()

                # Step 1: Compute class-specific mean loss curves
        mean_loss_by_class = {
            cls: losses[targets == cls].mean(axis=0)
            for cls in np.unique(targets)
        }

        # Step 2: Compute residual loss curve per sample
        residuals = np.zeros_like(losses)
        for i in range(len(losses)):
            residuals[i] = losses[i] - mean_loss_by_class[targets[i]]
        losses = residuals  
        # Combine targets and biases into string labels like "00", "01", "10", "11"
        group_labels = np.array([f"{t}{b}" for t, b in zip(targets, biases)])

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(losses)

        # Plot
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(group_labels)
        markers = ['o', 's', '^', 'D']  # One for each group

        for i, label in enumerate(np.array(["00", "11", "01", "10"])):
            idx = group_labels == label
            plt.scatter(
                tsne_results[idx, 0],
                tsne_results[idx, 1],
                marker=markers[i % len(markers)],
                edgecolors='w',
                linewidths=0.5,
                alpha=0.5,
                label=f"Target={label[0]}, Bias={label[1]}"
            )

        plt.legend(title="Target / Bias Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("t-SNE of Per-Sample Loss Vectors (5 epochs)")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()
        plt.savefig("tmp.png")
        plt.show()

        return

    def plot_mean_loss_curves(self):
        biases = []
        targets = []
        losses = []

        for batch in tqdm(self.dataloaders["train"]):
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]

            batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
            batch_losses = torch.stack(batch_losses)
            losses.append(batch_losses)

        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        losses = torch.cat(losses).numpy()  # (N, 5)

        mean_loss_by_class = {
            cls: losses[targets == cls].mean(axis=0)
            for cls in np.unique(targets)
        }

        # Combine into groups "00", "01", "10", "11"
        group_labels = np.array([f"{t}{b}" for t, b in zip(targets, biases)])

        # Prepare plot
        plt.figure(figsize=(10, 6))
        unique_groups = ["00", "01", "10", "11"]
        colors = ['blue', 'orange', 'green', 'red']

        for group, color in zip(unique_groups, colors):
            mask = group_labels == group
            if np.sum(mask) == 0:
                continue  # skip empty groups

            mean_curve = losses[mask].mean(axis=0)
            plt.plot(range(1, 51), mean_curve, label=f"Target={group[0]}, Bias={group[1]}", color=color, linewidth=2)
        # plt.plot(range(1, 51), mean_loss_by_class[0], label=f"m Target=0", color='black', linewidth=2)
        # plt.plot(range(1, 51), mean_loss_by_class[1], label=f"m Target=1", color='yellow', linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.title("Mean Loss Curve per Subgroup")
        plt.legend(title="Target / Bias Group")
        plt.grid(True)
        plt.xticks(range(1, 51))
        plt.tight_layout()
        plt.savefig("tmp_mean_loss_curves.png")
        plt.show()
        return 

    def plot_individual_loss_curves_norm(self, ma_window=5):
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w

        biases = []
        targets = []
        losses = []

        for batch in tqdm(self.dataloaders["train"]):
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]

            batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
            batch_losses = torch.stack(batch_losses)
            losses.append(batch_losses)

        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        losses = torch.cat(losses).numpy()  # (N, T)

        T = losses.shape[1]
        mean_loss_by_class = {
            cls: losses[targets == cls].mean(axis=0)
            for cls in np.unique(targets)
        }

        group_labels = np.array([f"{t}{b}" for t, b in zip(targets, biases)])

        plt.figure(figsize=(12, 6))
        unique_groups = ["00", "01", "10", "11"]
        colors = ['blue', 'orange', 'green', 'red']

        for group, color in zip(unique_groups, colors):
            mask = group_labels == group
            if np.sum(mask) == 0:
                continue

            target_class = int(group[0])
            class_mean = mean_loss_by_class[target_class]

            indices = np.where(mask)[0]
            if len(indices) > 100:
                indices = np.random.choice(indices, size=100, replace=False)

            for i in indices:
                normalized_curve = losses[i] - class_mean
                smoothed = moving_average(normalized_curve, ma_window)
                plt.plot(
                    range(1, len(smoothed) + 1),
                    smoothed,
                    color=color,
                    alpha=0.3
                )

        for group, color in zip(unique_groups, colors):
            plt.plot([], [], color=color, label=f"Target={group[0]}, Bias={group[1]}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss (Sample - Mean Class Loss)")
        plt.title(f"Smoothed Normalized Loss Curves per Subgroup (100 samples each, MA={ma_window})")
        plt.legend(title="Target / Bias Group")
        plt.grid(True)
        plt.xticks(range(1, T + 1, 5))
        plt.tight_layout()
        plt.savefig("tmp_individual_loss_curves_normalized_smooth.png")
        plt.show()
        return

    def plot_mean_loss_curves_norm(self):
        biases = []
        targets = []
        losses = []

        for batch in tqdm(self.dataloaders["train"]):
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]

            batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
            batch_losses = torch.stack(batch_losses)
            losses.append(batch_losses)

        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        losses = torch.cat(losses).numpy()  # (N, 5)

        mean_loss_by_class = {
            cls: losses[targets == cls].mean(axis=0)
            for cls in np.unique(targets)
        }

        # Combine into groups "00", "01", "10", "11"
        group_labels = np.array([f"{t}{b}" for t, b in zip(targets, biases)])

        # Prepare plot
        plt.figure(figsize=(10, 6))
        unique_groups = ["00", "01", "10", "11"]
        colors = ['blue', 'orange', 'green', 'red']

        for group, color in zip(unique_groups, colors):
            mask = group_labels == group
            if group == "00" or group == "01":
                mean_l = mean_loss_by_class[0]
            else:
                mean_l = mean_loss_by_class[1]
            if np.sum(mask) == 0:
                continue  # skip empty groups
            std = losses[mask].std(axis=0)
            mean_curve = losses[mask].mean(axis=0) - mean_l
            plt.plot(range(1, 51), mean_curve, label=f"Target={group[0]}, Bias={group[1]}", color=color, linewidth=2)
        # plt.plot(range(1, 51), mean_loss_by_class[0], label=f"m Target=0", color='black', linewidth=2)
        # plt.plot(range(1, 51), mean_loss_by_class[1], label=f"m Target=1", color='yellow', linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.title("Mean Loss Curve per Subgroup")
        plt.legend(title="Target / Bias Group")
        plt.grid(True)
        plt.xticks(range(1, 51))
        plt.tight_layout()
        plt.savefig("tmp_mean_loss_curves_normalized.png")
        plt.show()
        return 


    def timeseries_classification(self):
        print("timeseries_classification")
        biases = []
        targets = []
        losses = []
        indices_all = []
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w
        # Gather all losses, targets, and biases
        for batch in tqdm(self.dataloaders["train"]):
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]
            indices_all.append(indices)

            batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
            batch_losses = torch.stack(batch_losses)
            losses.append(batch_losses)

        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        losses = torch.cat(losses).numpy()  # (N, T)
        indices_all = torch.cat(indices_all).numpy()
        save_path = "timeseries_data.npz"
        np.savez(save_path, biases=biases, targets=targets, losses=losses)

        # train_lstm_classifier(losses=losses, targets=targets, biases=biases)
        from pyts.classification import KNeighborsClassifier
        from pyts.datasets import load_gunpoint
        X_train, X_test, y_train, y_test = train_test_split(losses, targets != biases, test_size=0.2, random_state=42, stratify=targets != biases)
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Print accuracy
        print("Accuracy:", clf.score(X_test, y_test))

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print("Confusion Matrix:")
        print(cm)

    def infer_bias_from_loss_trend(self):
        biases = []
        targets = []
        losses = []
        indices_all = []
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w
        # Gather all losses, targets, and biases
        for batch in tqdm(self.dataloaders["train"]):
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]
            indices_all.append(indices)

            batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
            batch_losses = torch.stack(batch_losses)
            losses.append(batch_losses)

        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        losses = torch.cat(losses).numpy()  # (N, T)
        indices_all = torch.cat(indices_all).numpy()

        # train_lstm_classifier(losses=losses, targets=targets, biases=biases)


        num_epochs = losses.shape[1]

        # Step 1: Compute class-specific mean loss curves
        mean_loss_by_class = {
            cls: losses[targets == cls].mean(axis=0)
            for cls in np.unique(targets)
        }

        # Step 2: Compute residual loss curve per sample
        residuals = []

        for i in range(len(losses)):
            l = moving_average(losses[i], 5)
            m = moving_average(mean_loss_by_class[targets[i]], 5)
            # residual = losses[i] - mean_loss_by_class[targets[i]]
            
            residuals.append(l - m)

        residuals = np.stack(residuals)  # Shape (N, T - window + 1)
        self.preproc_losses = residuals
        self.losses = {}
        for idx, loss_val in zip(indices_all, residuals):
            idx_item = idx
            if idx_item not in self.losses:
                self.losses[idx_item] = []
            self.losses[idx_item].append(loss_val)
        # Step 3: Define a bias signal: trend of residuals
        # Option 1: difference between last and first epoch residual
        residual_trend2 = residuals.sum(axis=1)
        residual_trend = residuals.std(axis=1)
        print(residuals[biases==targets].mean(), residuals[biases==targets].std())
        print(residuals[biases!=targets].mean(), residuals[biases!=targets].std())


        # residual_slopes = np.array([linregress(np.arange(50), r).slope for r in residuals])
        # residual_auc = residuals.sum(axis=1)
        # residual_trend = residual_slopes + 0.1 * residual_auc

    # def preproc_losses(self):
    #     biases = []
    #     targets = []
    #     losses = []
    #     indices_all = []
    #     def moving_average(x, w):
    #         return np.convolve(x, np.ones(w), 'valid') / w
    #     # Gather all losses, targets, and biases
    #     for batch in tqdm(self.dataloaders["train"]):
    #         biases.append(batch[self.biases[0]].cpu())
    #         targets.append(batch["targets"].cpu())
    #         indices = batch["index"]
    #         indices_all.append(indices)

    #         batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
    #         batch_losses = torch.stack(batch_losses)
    #         losses.append(batch_losses)

    #     biases = torch.cat(biases).numpy()  # (N,)
    #     targets = torch.cat(targets).numpy()  # (N,)
    #     losses = torch.cat(losses).numpy()  # (N, T)
    #     indices_all = torch.cat(indices_all).numpy()

    #     # train_lstm_classifier(losses=losses, targets=targets, biases=biases)


    #     num_epochs = losses.shape[1]

    #     # Step 1: Compute class-specific mean loss curves
    #     mean_loss_by_class = {
    #         cls: losses[targets == cls].mean(axis=0)
    #         for cls in np.unique(targets)
    #     }

    #     # Step 2: Compute residual loss curve per sample
    #     residuals = []

    #     for i in range(len(losses)):
    #         # l = moving_average(losses[i], 5)
    #         # m = moving_average(mean_loss_by_class[targets[i]], 5)
    #         residual = losses[i] - mean_loss_by_class[targets[i]]
            
    #         residuals.append(residual)

    #     residuals = np.stack(residuals)  # Shape (N, T - window + 1)
    #     self.losses = {}
    #     for idx, loss_val in zip(indices_all, residuals):
    #         idx_item = idx
    #         self.losses[idx_item] = [x for x in loss_val]
        
    #     return 
    
    def collect_losses_preproc(self): 
        self.model.eval()
        biases_all = []
        targets_all = []
        losses_all = []
        indices_all = []
        with torch.no_grad():
            for batch in tqdm(self.dataloaders["train"]):
                bsz = batch["targets"].shape[0]
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                biases = batch[self.biases[0]]
                indices = batch["index"]

                biases_all.append(batch[self.biases[0]].cpu())
                targets_all.append(batch["targets"].cpu())
                indices_all.append(indices)

                
                outputs = self.model(inputs)
                if not isinstance(outputs, tuple):
                    raise ValueError("Model output must be a tuple (logits, features)")
                outputs, features = outputs
                logits = outputs
                # logits = self.metric_loss_default.predict(features,targets)
                loss = F.cross_entropy(logits, targets, reduction="none")
                losses_all.append(loss.detach().cpu())
                
            biases = torch.cat(biases_all).cpu().numpy()  # (N,)
            targets = torch.cat(targets_all).cpu().numpy()  # (N,)
            losses = torch.cat(losses_all).cpu().numpy()  # (N, T)
            indices_all = torch.cat(indices_all).cpu().numpy()
            mean_loss_by_class = {
                cls: losses[targets == cls].mean(axis=0)
                for cls in np.unique(targets)
            }

            # Step 2: Compute residual loss curve per sample
            residuals = []

            for i in range(len(losses)):
                # l = moving_average(losses[i], 5)
                # m = moving_average(mean_loss_by_class[targets[i]], 5)
                residual = losses[i] - mean_loss_by_class[targets[i]]
                
                residuals.append(residual)

            residuals = np.stack(residuals)  # Shape (N, T - window + 1)
            # self.losses = {}
            # for idx, loss_val in zip(indices_all, residuals):
            #     idx_item = idx
            #     self.losses[idx_item] = [loss_val.item()]

            for idx, loss_val in zip(indices_all, residuals):
                idx_item = idx
                if idx_item not in self.losses:
                    self.losses[idx_item] = []
                self.losses[idx_item].append(loss_val)
            # print(len(self.losses))
            # print(len(indices_all))
        self.model.train()

    def collect_losses(self): 
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders["train"]):
                bsz = batch["targets"].shape[0]
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                indices = batch["index"]
                outputs = self.model(inputs)
                if not isinstance(outputs, tuple):
                    raise ValueError("Model output must be a tuple (logits, features)")
                outputs, features = outputs

                logits = self.metric_loss_default.predict(features,targets)
                loss = F.cross_entropy(logits, targets, reduction="none")
                
                # self.losses = {**self.losses, **{idx.item() : loss[i].item() for idx, i in zip(indices,range(targets.shape[0]))} }
                for idx, loss_val in zip(indices, loss):
                    idx_item = idx.item()
                    if idx_item not in self.losses:
                        self.losses[idx_item] = []
                    self.losses[idx_item].append(loss_val.item())
        self.model.train()

    def _method_specific_setups(self):
        self.losses = {}
        self.preproc_losses = {}
        # self.bias_detection()

    def bias_detection(self):
        cfg = self.cfg
        self.model.train()
        

       
        LOSSES_SAVE_PATH = os.path.join(self.log_path, "losses_50ep")
        if os.path.exists(LOSSES_SAVE_PATH):
            print("Loading precomputed losses...")
            self.losses = torch.load(LOSSES_SAVE_PATH)
            
        else:
        # if True:
            detection_optimizer = torch.optim.SGD(
            self.model.parameters(),
            cfg.MITIGATOR.ARC.DETECTION_OPTIMIZER.LR,
            momentum=cfg.MITIGATOR.ARC.DETECTION_OPTIMIZER.MOMENTUM,
            weight_decay=cfg.MITIGATOR.ARC.DETECTION_OPTIMIZER.WEIGHT_DECAY,
        )
            
            print(
                f"training for {cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS} epoch(s) to detect the biases."
            )
            for epoch in range(cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS):
                total_loss = 0.0
                correct = 0
                total = 0

                for batch in self.dataloaders["train"]:
                    detection_optimizer.zero_grad()
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    biases = batch[self.biases[0]].to(self.device)

                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, features = outputs
                    outputs = self.metric_loss_default.forward_def(features,targets,biases)
                    loss = self.criterion(outputs, targets)

                    # Backpropagation
                    
                    self._loss_backward(loss)
                    detection_optimizer.step()
                    

                    # Track loss
                    total_loss += loss.item()

                    # Track accuracy
                    predicted = outputs.argmax(dim=1)  # Assuming classification task
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

                avg_loss = total_loss / len(self.dataloaders["train"])
                accuracy = correct / total * 100 if total > 0 else 0

                print(f"Epoch [{epoch+1}/{cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                self.collect_losses()

            torch.save(self.losses, LOSSES_SAVE_PATH)
            print(f"Losses saved at {LOSSES_SAVE_PATH}")
        
        # self.plot_mean_loss_curves()
        # self.plot_mean_loss_curves_norm()
        # self.plot_individual_loss_curves_norm()
        # self.preproc_losses()
        # # self.visualize_losses()
        # self.compute_mean_and_normalize()
        self.timeseries_classification()
        # self.compute_losses_performance()
        self._setup_models()
        self._setup_optimizer()

    def compute_mean_and_normalize(self):
        """
        1) Computes the mean loss for each key in self.losses.
        2) Normalizes these mean losses to [0, 1] range.
        3) Returns (mean_losses, normed_mean_losses).
        """
        # Step 1: Compute the mean loss for each key
        mean_losses = {}
        for key, loss_list in self.losses.items():
            # print(loss_list)
            mean_losses[key] = sum(loss_list) / len(loss_list)
        
        
        # Step 2: Normalize mean losses to [0,1]
        min_val = min(mean_losses.values())
        max_val = max(mean_losses.values())
        
        range_val = max_val - min_val

        # Target range
        a, b = 0.01 , 1.0  # new min and max

        # Apply normalization: x' = a + ((x - min) * (b - a)) / (max - min)
        normed_mean_losses = {
            k: 1 - (a + ((v - min_val) * (b - a) / range_val))
            for k, v in mean_losses.items()
}
        self.loss_th = np.mean(list(normed_mean_losses.values())) # np.mean(list(normed_mean_losses.values())) + np.std(list(normed_mean_losses.values())) 
        # print(self.loss_th, np.min(list(normed_mean_losses.values())), np.max(list(normed_mean_losses.values())))
        self.preproc_losses = normed_mean_losses
        # print(len(self.preproc_losses))

        return 
    
    def compute_losses_performance(self):
        all_preds = []
        all_labels = [] 
        for batch in tqdm(self.dataloaders["train"]):
            bsz = batch["targets"].shape[0]
            inputs = batch["inputs"]
            targets = batch["targets"]
            indices = batch["index"]
            biases = batch[self.biases[0]]

            loss_pre = torch.tensor([self.preproc_losses[idx.item()] for idx in indices])
            bc_wrt_losses = loss_pre > self.loss_th
            bc_wrt_labels = targets != biases
            all_preds.append(bc_wrt_losses.cpu())
            all_labels.append(bc_wrt_labels.cpu())

        # Stack predictions and labels
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Compute metrics
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Loss-based classification of bias conflict:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Confusion matrix (rows: true, cols: pred):\n{cm}")
        

    def plot_val_tsne(self, val_loader, num_samples=1000, perplexity=30, random_state=42):
        self.model.eval()
        
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    _, features = outputs
                else:
                    raise ValueError("Model must return (logits, features) tuple")

                all_features.append(features.detach().cpu())
                all_labels.append(targets.detach().cpu())

                # Optional: limit the number of samples to speed up t-SNE
                if len(torch.cat(all_labels)) >= num_samples:
                    break

        features = torch.cat(all_features)[:num_samples]
        labels = torch.cat(all_labels)[:num_samples]

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        embeddings = tsne.fit_transform(features.numpy())

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
        plt.title("t-SNE of Validation Set Features")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.savefig("tmp_sa.png")
        plt.show()
