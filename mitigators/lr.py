import math
import os
from tqdm import tqdm
import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc
import torch.nn.functional as F
from scipy.stats import linregress
import gc  # For garbage collection

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
from .losses import FocalLoss, LDAMLoss, MAELoss, NCELoss, NCEandAGCE, NCEandAUE, RCELoss, SubcenterArcMarginProduct, LabelSmoothSoftmaxCEV1, ArcMarginProduct, WBLoss
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
from pyts.classification import *
from pyts.datasets import load_gunpoint

def get_grad_for_sample(index):
    """
    Retrieves the flattened gradient for a specific training sample by index.

    Args:
        index (int): The unique ID or index of the sample.

    Returns:
        A 1D tensor representing the flattened gradient of the sample.
    """
    return per_sample_grads.get(index)

def flatten_grads(model):
    """
    Flattens the gradients of all model parameters into a single 1D tensor.
    Assumes that backward() has already been called.

    Returns:
        A 1D tensor containing the concatenated gradients of all parameters.
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().view(-1))
        else:
            # If grad is None (e.g., for frozen layers), append zeros of same shape
            grads.append(torch.zeros_like(param).view(-1))
    return torch.cat(grads)

def plot_margin_histogram(margins, targets=None, biases=None, save_path="margin_histogram.png"):
    """
    Args:
        margins: torch.Tensor of shape [N] — learnable margins
        targets: numpy or torch array [N] — ground truth labels (0 or 1)
        biases: numpy or torch array [N] — bias labels (0 or 1)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(biases, torch.Tensor):
        biases = biases.cpu().numpy()
    if isinstance(margins, torch.Tensor):
        margins = margins.detach().cpu().numpy()

    group_labels = ['00', '01', '10', '11']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(8, 6))

    for label, color in zip(group_labels, colors):
        t, b = int(label[0]), int(label[1])
        group_mask = (targets == t) & (biases == b)
        group_margins = margins[group_mask]
        plt.hist(group_margins, bins=30, alpha=0.5, label=f'Group {label}', color=color)
    # plt.hist(margins, bins=30, alpha=0.5)

    plt.xlabel("Margin Value")
    plt.ylabel("Count")
    plt.title("Histogram of Learnable Margins by (Target, Bias) Group")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved margin histogram to {save_path}")


class LRTrainer(BaseTrainer):
    def _setup_resume(self):
        return
    
    def _setup_criterion(self):
        # self.criterion = LabelSmoothSoftmaxCEV1() 
        self.criterion = torch.nn.CrossEntropyLoss() 
        # self.criterion = NCEandAGCE(num_classes=2)
        # self.criterion = MAELoss(num_classes=2)
        # self.criterion = FocalLoss(gamma=5)
        
        self.metric_loss = ArcMarginProduct(512,2,m=0.5,m_b=0.5)

        self.metric_loss.to(self.device)


    def _train_iter(self, batch):
        self.optimizer.zero_grad()
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)
        indices = batch["index"].to(self.device)
        # mask = biases == targets 
        # inputs = inputs[mask]
        # targets = targets[mask]
        # biases = biases[mask]
        # indices = indices[mask]
        
        outputs = self.model(inputs)
        if not isinstance(outputs, tuple):
            raise ValueError("Model output must be a tuple (logits, features)")
        outputs, features = outputs

        # margins = self.margins[indices].to(outputs.dtype).to(self.device)
        # print(torch.max(margins), torch.min(margins))

        # logits = self.metric_loss.forward(features,targets, margins)

        # probs = torch.softmax(logits, dim=1)
        # preds = torch.argmax(probs, dim=1)

        # # Find misclassified samples
        # misclassified = preds != targets

        # # Compute confidence of the predicted class
        # confidences = probs[torch.arange(len(probs)), preds]
        # # print(confidences[misclassified])
        # # Define a threshold for "high confidence" (e.g., 0.7 or 0.9)
        # conf_threshold = 0.99

        # # High-confidence misclassified mask
        # hc_misclassified = misclassified & (confidences > conf_threshold)

        # num_hc_misclassified = hc_misclassified.sum().item()
        # print(f"High-confidence misclassified samples: {num_hc_misclassified}")




        # preds = torch.argmax(logits, dim=1)
        # correct = preds != targets
        # margins[hc_misclassified] = 0.2
        # self.margins[indices] = margins.detach().cpu()
        logits = outputs
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
        # logits = self.metric_loss.predict(features,targets)
        logits = outputs
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
        # plot_margin_histogram(self.margins,self.full_targets,self.full_biases)
        return avg_loss
    

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

                # logits = self.metric_loss.predict(features,targets)
                logits = outputs
                loss = F.cross_entropy(logits, targets, reduction="none")
                
                # self.losses = {**self.losses, **{idx.item() : loss[i].item() for idx, i in zip(indices,range(targets.shape[0]))} }
                for idx, loss_val in zip(indices, loss):
                    idx_item = idx.item()
                    if idx_item not in self.losses:
                        self.losses[idx_item] = []
                    self.losses[idx_item].append(loss_val.item())
        self.model.train()


    def timeseries_classification(self):
        self.model.eval()
        print("timeseries_classification")
        biases = []
        targets = []
        feats = []
        losses = []
        indices_all = []

        # Gather all losses, targets, and biases
        for batch in tqdm(self.dataloaders["train"]):
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]
            indices_all.append(indices)
            inputs = batch["inputs"].to(self.device)
            labels = batch["targets"]
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                _, features = outputs
            else:
                raise ValueError("Model must return (logits, features)")
            feats.append(features.detach().cpu())
            

            batch_losses = [torch.tensor(self.losses[idx.item()]) for idx in indices]
            batch_losses = torch.stack(batch_losses)
            losses.append(batch_losses)

        feats = torch.cat(feats).numpy()
        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        losses = torch.cat(losses).numpy()  # (N, T)
        indices_all = torch.cat(indices_all).numpy()
        save_path = "timeseries_data_feats.npz"
        np.savez(save_path, biases=biases, targets=targets, losses=losses, feats=feats, indices_all=indices_all)
        
        # Labels: 1 if bias-conflicting
        true_labels = (targets != biases).astype(int)

        # Train/test split on losses (timeseries classifier input)
        X_train, X_test, y_train, y_test, feats_train, feats_test, idx_train, idx_test = train_test_split(
            losses, true_labels, feats, indices_all, test_size=0.2, random_state=42, stratify=true_labels
        )

        # Train a time series classifier (e.g., BOSSVS)
        # clf = BOSSVS(window_size=28)
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
        feats_bc = feats_test[mask_bc_pred]
        targets_bc = targets[idx_test[mask_bc_pred]]
        biases_bc = biases[idx_test[mask_bc_pred]]

        # Optional: subsample for clarity
        max_plot = 1000
        if feats_bc.shape[0] > max_plot:
            sel = np.random.choice(feats_bc.shape[0], max_plot, replace=False)
            feats_bc = feats_bc[sel]
            targets_bc = targets_bc[sel]
            biases_bc = biases_bc[sel]

        # Group labels as strings: '00', '01', etc.
        group_labels = np.array([f"{t}{b}" for t, b in zip(targets_bc, biases_bc)])

        print(f"Running t-SNE on {len(feats_bc)} predicted bias-conflicting samples...")
        tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
        tsne_proj = tsne.fit_transform(feats_bc)

        # Unique groups and color mapping
        unique_groups = sorted(set(group_labels))
        colors = ['blue', 'orange', 'green', 'red']
        group_color_map = {grp: colors[i] for i, grp in enumerate(unique_groups)}

        plt.figure(figsize=(8, 6))
        for grp in unique_groups:
            idxs = np.where(group_labels == grp)[0]
            plt.scatter(
                tsne_proj[idxs, 0], tsne_proj[idxs, 1],
                label=f'Group {grp}',
                color=group_color_map[grp],
                s=10, alpha=0.6
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


    def _method_specific_setups(self):
        self.losses = {}
        # self.bias_detection()
        # self.timeseries_classification()
        # print(fa)
        num_samples = len(self.sets["train"])  # total number of training samples
        self.margins = torch.zeros(num_samples)  # or init with some constant
        # self.margins = nn.Parameter(torch.empty(num_samples).uniform_(-0.5, 0.5))

        # self.optimizer = torch.optim.Adam(
        #     list(self.model.parameters()) + [self.margins],
        #     self.cfg.MITIGATOR.ARC.DETECTION_OPTIMIZER.LR,
        #     # momentum=cfg.MITIGATOR.ARC.DETECTION_OPTIMIZER.MOMENTUM,
        #     weight_decay=self.cfg.MITIGATOR.ARC.DETECTION_OPTIMIZER.WEIGHT_DECAY,
        # )
        self.full_targets = []
        self.full_biases = []

        for item in self.sets["train"]:
            # print(item["index"])
            self.full_targets.append(item["targets"])
            self.full_biases.append(item[self.biases[0]])  # use the actual key from your dataset

        self.full_targets = torch.tensor(self.full_targets)
        self.full_biases = torch.tensor(self.full_biases)



        self.bias_detection2()
        # self.timeseries_classification()
        print(fa)
    def bias_detection(self):
        cfg = self.cfg
        self.model.train()
        

       
        LOSSES_SAVE_PATH = os.path.join(self.log_path, "losses")
        # if os.path.exists(LOSSES_SAVE_PATH):
        #     print("Loading precomputed losses...")
        #     self.losses = torch.load(LOSSES_SAVE_PATH)
            
        # else:
        if True:
            detection_optimizer = torch.optim.Adam(
            self.model.parameters(),
            cfg.SOLVER.LR,
            # momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            self.scheduler_bd = torch.optim.lr_scheduler.MultiStepLR(
                detection_optimizer,
                milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
            print(
                f"training for {cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS} epoch(s) to detect the biases."
            )
            for epoch in range(cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS):
                self.model.train()
                total_loss = 0.0
                correct = 0
                total = 0

                for batch in self.dataloaders["train"]:
                    detection_optimizer.zero_grad()
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    indices = batch["index"]
                    biases = batch[self.biases[0]].to(self.device)
                       
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, features = outputs
                        
                    # logits = self.metric_loss.forward_def(features,targets)
                    logits = outputs
                    loss = self.criterion(logits, targets)

                    # Backpropagation
                    
                    self._loss_backward(loss)
                    detection_optimizer.step()
                   
                    # print(torch.max(margins), torch.min(margins))
                    # Track loss
                    total_loss += loss.item()

                    # Track accuracy
                    predicted = outputs.argmax(dim=1)  # Assuming classification task
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
                self.scheduler_bd.step()
                avg_loss = total_loss / len(self.dataloaders["train"])
                accuracy = correct / total * 100 if total > 0 else 0

                print(f"Epoch [{epoch+1}/{cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                self.collect_losses()
                train_performance = self._validate_epoch(stage="train")

            torch.save(self.losses, LOSSES_SAVE_PATH)
            print(f"Losses saved at {LOSSES_SAVE_PATH}")
            self._save_checkpoint(tag="bias_detection_model")
            print(f"Model saved at bias_detection_model")
        
        # self.plot_mean_loss_curves()
        # self.plot_mean_loss_curves_norm()
        # self.plot_individual_loss_curves_norm()
        # # self.visualize_losses()
        # self.compute_mean_and_normalize()
        # self.timeseries_classification()
        # self.compute_losses_performance()
        # self._setup_models()
        # self._setup_optimizer()


    def bias_detection2(self):
        # Initialize tracking dictionaries
        gas_scores = defaultdict(list)  # Gradient Agreement Score
        confidence_scores = defaultdict(list)  # Softmax confidence over time
        lds_scores = {}  # Learning Delay Score
        cos_scores = {}  # Confidence Oscillation Score

        conf_threshold = 0.99  # Confidence threshold for LDS
        final_scores = {}
        top_k = 100

        self.model.train()
        cfg = self.cfg
        print(
                f"training for {cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS} epoch(s) to detect the biases."
            )
        detection_optimizer = torch.optim.Adam(
            self.model.parameters(),
            cfg.SOLVER.LR,
            # momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        self.scheduler_bd = torch.optim.lr_scheduler.MultiStepLR(
            detection_optimizer,
            milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
            gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
        )
        
        for epoch in range(cfg.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS):
            per_sample_grads = {}

            for batch in tqdm(self.dataloaders["train"]):
                
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                indices = batch["index"]
                biases = batch[self.biases[0]].to(self.device)
                    
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs, features = outputs
                loss = F.cross_entropy(outputs, targets)
                
                loss.backward()

                # Compute per-sample gradients
                # for i in range(len(inputs)):
                #     self.model.zero_grad()
                #     loss[i].backward(retain_graph=True)
                #     g_i = flatten_grads(self.model)  # custom: flattens current sample's grads
                #     per_sample_grads[indices[i].item()] = g_i.clone().detach()

                #     # Accumulate batch gradients
                #     if i == 0:
                #         g_batch = g_i.clone()
                #     else:
                #         g_batch += g_i
                # g_batch /= len(inputs)

                # Compute and store GAS and Confidence
                for i in range(len(inputs)):
                    # g_i = per_sample_grads.get(indices[i].item())  # get saved gradient for sample i
                    # sim = cosine_similarity(g_i, g_batch)
                    # sim = F.cosine_similarity(g_i.unsqueeze(0), g_batch.unsqueeze(0), dim=1).item()
                    # gas_scores[indices[i]].append(sim)

                    # conf = softmax(outputs[i])[targets[i]].item()
                    conf = F.softmax(outputs[i], dim=0)[targets[i]].item()
                    confidence_scores[indices[i].item()].append(conf)
                    # print(confidence_scores)

                # Optional: reset gradients after each epoch
                detection_optimizer.step()
                detection_optimizer.zero_grad()

                # # Clean up GPU memory
                # del inputs, targets, biases, outputs, loss, g_batch, g_i
                # torch.cuda.empty_cache()
                # gc.collect()

        # After training — compute LDS and COS
        for idx, conf_list in confidence_scores.items():
            # LDS: First epoch where confidence > threshold
            lds_scores[idx] = next((i for i, c in enumerate(conf_list) if c >= conf_threshold), len(conf_list))
            # print(conf_list, np.std(conf_list))
            # COS: Std deviation of confidence
            cos_scores[idx] = np.std(conf_list)

        # Compute final combined score
        alpha, beta, gamma = 1.0, 1.0, 0.0  # Hyperparameters to tune
        avg_gas_d = {}
        for idx in confidence_scores:
            # print(idx)
            # avg_gas = np.mean(gas_scores[idx])
            # avg_gas_d[idx] = avg_gas
            score = beta* lds_scores[idx] + gamma * cos_scores[idx]
            final_scores[idx] = score
            # print(cos_scores[idx])
        # Sort by final_scores — top-k are likely bias-conflicting
        bias_conflicting_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]


        biases = []
        targets = []
        indices_all = []

        # Gather all losses, targets, and biases
        for batch in self.dataloaders["train"]:
            biases.append(batch[self.biases[0]].cpu())
            targets.append(batch["targets"].cpu())
            indices = batch["index"]
            indices_all.append(indices)
          

        biases = torch.cat(biases).numpy()  # (N,)
        targets = torch.cat(targets).numpy()  # (N,)
        indices_all = torch.cat(indices_all).numpy()

        # Group final scores into BA (target == bias) and BC (target != bias)
        scores_ba = []  # bias-aligned
        scores_bc = []  # bias-conflicting

        for idx, score in final_scores.items():
            # Get target and bias for the sample
            # sample_idx = np.where(indices_all == idx)[0][0]
            target = targets[indices_all[idx]]
            bias = biases[indices_all[idx]]

            if target == bias:
                scores_ba.append(score)
            else:
                scores_bc.append(score)

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(scores_ba, bins=30, alpha=0.6, label="BA (target == bias)")
        plt.hist(scores_bc, bins=30, alpha=0.6, label="BC (target != bias)")

        plt.xlabel("Final Bias Conflict Score")
        plt.ylabel("Number of Samples")
        plt.title("Histogram of Bias Conflict Scores by Sample Type")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        plt.savefig("bias_conflict_score_hist.png")
        plt.close()


        
        from scipy.stats import pearsonr
        from collections import defaultdict

        # Step 1: Group indices by target class
        class_to_indices = defaultdict(list)
        for idx in confidence_scores:
            sample_idx = np.where(indices_all == idx)[0][0]
            cls = targets[sample_idx]
            class_to_indices[cls].append(idx)

        negatively_correlated_pairs = []
        involved_indices = []

        # Step 2: For each class, find anti-correlated confidence trajectories
        for cls, indices_in_class in class_to_indices.items():
            for i in range(len(indices_in_class)):
                for j in range(i + 1, len(indices_in_class)):
                    idx_i = indices_in_class[i]
                    idx_j = indices_in_class[j]

                    ci = confidence_scores[idx_i]
                    cj = confidence_scores[idx_j]

                    if len(ci) == len(cj):
                        corr, _ = pearsonr(ci, cj)
                        if corr < -0.1:  # Strong negative correlation
                            negatively_correlated_pairs.append((idx_i, idx_j))
                            involved_indices.append([idx_i, idx_j])

        # Step 3: Count BA and BC for involved samples
        ba_count = 0
        bc_count = 0
        for idx in involved_indices:
            sample_idx = np.where(indices_all == idx)[0][0]
            target = targets[sample_idx]
            bias = biases[sample_idx]
            if target == bias:
                ba_count += 1
            else:
                bc_count += 1

        # Step 4: Print results
        print(f"Number of negatively correlated pairs: {len(negatively_correlated_pairs)}")
        print(f"Number of involved samples: {len(involved_indices)}")
        print(f"Bias-aligned (BA): {ba_count}")
        print(f"Bias-conflicting (BC): {bc_count}")


        save_path = "data.npz"
        np.savez(save_path, biases=biases, targets=targets, indices_all=indices_all)
        torch.save("avg_gas", avg_gas_d)
        torch.save("lds", lds_scores)
        torch.save("cos", cos_scores)



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
