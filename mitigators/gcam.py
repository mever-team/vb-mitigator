from tqdm import tqdm
import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc
import torch.nn.functional as F

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
import torch.nn as nn
import itertools
import torch.autograd as autograd
from tools.utils import (
    log_msg,
    save_checkpoint,
    load_checkpoint,)
import os 
from torch.autograd import grad as torch_grad
from torch.autograd import Variable

def visualize_clusters(gradient_list, cluster_labels, targets, bias_labels, out_name="tmp.png", method='tsne'):
    X = np.vstack(gradient_list)

    # Dimensionality reduction
    if method == 'pca':
        X_2d = PCA(n_components=2).fit_transform(X)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        X_2d = TSNE(n_components=2).fit_transform(X)
    else:
        raise ValueError("Unknown method")

    plt.figure(figsize=(18, 5))

    # --- Plot 1: Clustering ---
    plt.subplot(1, 3, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0)  # Invisible scatter to set limits
    for i in range(X_2d.shape[0]):
        label = str(cluster_labels[i]) if cluster_labels[i] != -1 else "O"  # 'O' for outlier
        plt.text(X_2d[i, 0], X_2d[i, 1], label, fontsize=6, ha='center', va='center')
    plt.title("Clusters (DBSCAN)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # --- Plot 2: Ground Truth Targets ---
    plt.subplot(1, 3, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0)  # Invisible scatter
    for i in range(X_2d.shape[0]):
        plt.text(X_2d[i, 0], X_2d[i, 1], str(targets[i]), fontsize=6, ha='center', va='center')
    plt.title("Ground Truth Targets")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # --- Plot 3: Bias Labels ---
    plt.subplot(1, 3, 3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0)  # Invisible scatter
    for i in range(X_2d.shape[0]):
        plt.text(X_2d[i, 0], X_2d[i, 1], str(bias_labels[i]), fontsize=6, ha='center', va='center')
    plt.title("Bias Labels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.show()

def get_gradient_vector(model, loss):
    """Flatten all gradients into a single vector."""
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    grad_vector = torch.cat([g.contiguous().view(-1) for g in grads if g is not None])
    return grad_vector

def cosine_distance(vec1, vec2):
    """1 - cosine similarity to use as a regularization penalty."""
    cos_sim = F.cosine_similarity(vec1, vec2, dim=0, eps=1e-8)
    # print(cos_sim)
    return 1 - cos_sim

def project(a, b):
    """
    Project vector a onto vector b.

    Args:
        a (Tensor): Vector to project (e.g., g_ba)
        b (Tensor): Vector to project onto (e.g., g_bc)
    
    Returns:
        Tensor: The projection of a onto b
    """
    b_norm_sq = torch.dot(b, b) + 1e-8  # Avoid division by zero
    scalar_proj = torch.dot(a, b) / b_norm_sq
    proj = scalar_proj * b
    return proj

def get_per_sample_gradients(model, inputs, targets, loss_fn):
    
    grads = []
    predictions = []
    true_labels = []
    # model.eval()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for i in range(inputs.size(0)):
        
        model.zero_grad()
        input_i = inputs[i].unsqueeze(0)
        target_i = targets[i].unsqueeze(0)

        output, _ = model(input_i)
        loss = loss_fn(output, target_i)
        pred_label = output.argmax(dim=1).item()
        predictions.append(pred_label)
        true_labels.append(target_i.item())

        grad_vector = torch.autograd.grad(loss, trainable_params, retain_graph=True, create_graph=False)
        grad_vector = torch.cat([g.contiguous().view(-1) for g in grad_vector if g is not None])
        grads.append(grad_vector.detach().cpu().numpy())

def get_gradient_norm_loss(model, inputs, targets, loss_fn):
    grad_norms = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for i in range(inputs.size(0)):
        model.zero_grad()
        input_i = inputs[i].unsqueeze(0)
        target_i = targets[i].unsqueeze(0)

        output, _ = model(input_i)
        loss = loss_fn(output, target_i)

        grad_vector = torch.autograd.grad(loss, trainable_params, retain_graph=True, create_graph=False)
        grad_vector = torch.cat([g.view(-1) for g in grad_vector if g is not None])
        
        norm = torch.norm(grad_vector, p=2)  # L2 norm
        grad_norms.append(norm)

    grad_norms = torch.stack(grad_norms)
    return grad_norms.mean()

    # # Group gradients by class
    # class_grads = defaultdict(list)

    # for grad, label in zip(grads, true_labels):
    #     class_grads[label].append(grad)

    # # Compute average gradient for each class
    # avg_grad_per_class = {}

    # for label, grad_list in class_grads.items():
    #     avg_grad = np.mean(grad_list, axis=0)
    #     avg_grad_per_class[label] = avg_grad  # numpy array of shape (D,)

    # # Step 2: Project each gradient away from its class-average gradient
    # grads_projected = []

    # for grad, label in zip(grads, true_labels):
    #     v = avg_grad_per_class[label]  # bias direction = class-average grad
    #     v_norm_sq = np.dot(v, v)
    #     if v_norm_sq > 1e-12:  # avoid division by zero
    #         projection = (np.dot(grad, v) / v_norm_sq) * v
    #         g_proj = grad - projection
    #     else:
    #         g_proj = grad  # if avg grad is (close to) zero, skip projection
    #     grads_projected.append(g_proj)

        
    # # --- Accuracy Per Class ---
    # class_correct = defaultdict(int)
    # class_total = defaultdict(int)

    # for pred, true in zip(predictions, true_labels):
    #     class_total[true] += 1
    #     if pred == true:
    #         class_correct[true] += 1

    # print("Per-class accuracy:")
    # for cls in sorted(class_total.keys()):
    #     acc = class_correct[cls] / class_total[cls]
    #     print(f"  Class {cls}: {acc:.2%} ({class_correct[cls]}/{class_total[cls]})")

    return grads

def detect_gradient_outliers(grads, true_labels, bias_labels, z_thresh=1):
    grads = np.array(grads)
    # true_labels = np.array(true_labels)
    # bias_labels = np.array(bias_labels)

    outliers_per_class = defaultdict(list)

    for cls in np.unique(true_labels):
        idx = np.where(true_labels == cls)[0]
        class_grads = grads[idx]  # shape (N_class, D)

        # Mean gradient of the class
        mean_grad = np.mean(class_grads, axis=0)

        # Distance of each gradient to the mean
        distances = np.linalg.norm(class_grads - mean_grad, axis=1)

        # Z-score
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        z_scores = (distances - mean_dist) / (std_dist + 1e-12)

        # Detect outliers
        outlier_indices = idx[z_scores > z_thresh]

        # print(f"\nClass {cls} → {len(outlier_indices)} outliers (Z > {z_thresh}):")
        # for i in outlier_indices:
        #     print(f"  Index {i}: Target = {true_labels[i]}, Bias = {bias_labels[i]}")

        outliers_per_class[cls] = outlier_indices.tolist()

    return outliers_per_class

def cluster_gradients(gradient_list, eps=1000.0, min_samples=5):
    gradients = np.vstack(gradient_list)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(gradients)
    return clustering.labels_  # -1 means outlier


def compute_input_gradient(model, image, label):
    image = image.clone().detach().unsqueeze(0).requires_grad_(True)
    label = torch.tensor([label]).to(image.device)

    model.zero_grad()
    output, _ = model(image)
    loss = F.cross_entropy(output, label)
    loss.backward()

    # Gradient w.r.t. input
    grad = image.grad.data.squeeze().cpu()
    return grad

def normalize_heatmap(grad):
    grad_abs = grad.abs().sum(dim=0)  # Sum over channels
    grad_norm = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min() + 1e-8)
    return grad_norm.numpy()

def visualize_heatmap(heatmap, image, title="Gradient Heatmap"):
    plt.figure(figsize=(6, 6))
    image_np = image.permute(1, 2, 0).cpu().numpy()
    heatmap_color = plt.cm.jet(heatmap)[:, :, :3]

    overlay = heatmap_color
    overlay = np.clip(overlay, 0, 1)

    plt.imshow(overlay)
    plt.axis('off')
    plt.title(title)
    
    plt.savefig("gradcam.png")
    plt.show()

def analyze_bias_regions(model, aligned_data, conflict_data):
    model.eval()

    
    t_l = 0
    conflict_grads = []
    for img, label in zip(conflict_data[0],conflict_data[1]):
        grad = compute_input_gradient(model, img, label)
        conflict_grads.append(grad)
        t_l = label
        print(t_l)
        break

    aligned_grads = []
    for img, label in zip(aligned_data[0],aligned_data[1]):
        if label!=t_l:
            continue
        grad = compute_input_gradient(model, img, label)
        aligned_grads.append(grad)
        break

    avg_aligned_grad = torch.stack(aligned_grads).mean(dim=0)
    avg_conflict_grad = torch.stack(conflict_grads).mean(dim=0)

    grad_diff = avg_aligned_grad- avg_conflict_grad
    heatmap = normalize_heatmap(grad_diff)

    # Visualize on one sample image (for context)
    sample_image = aligned_data[0][0].cpu()
    visualize_heatmap(heatmap, sample_image, title="Bias-Relevant Regions")

    return heatmap, avg_aligned_grad, avg_conflict_grad


def get_gradients(model, inputs, targets):
    model.zero_grad()
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    grads = torch.autograd.grad(
        loss, model.parameters(), retain_graph=True, create_graph=True
    )
    grads = torch.cat([g.view(-1) for g in grads if g is not None])
    return grads

def gradient_similarity_loss(grad_ba, grad_bc):
    # Cosine similarity (closer to 1 = more aligned)
    cos_sim = F.cosine_similarity(grad_ba.unsqueeze(0), grad_bc.unsqueeze(0))
    return 1 - cos_sim  # loss is low when gradients are aligned


    
class GCamTrainer(BaseTrainer):

    def _setup_resume(self):
        return


    def _set_train(self):
        super()._set_train()
        # Freeze all BN layers
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
    
    def _train_iter(self, batch):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)

        batch_size = targets.shape[0]

        # ba_mask = targets == biases 
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss_cl = loss_fn(outputs, targets)
        loss_cl_mean = loss_cl.mean()
        grad_norms = []

        for i in range(inputs.size(0)):
            grad_i = torch.autograd.grad(
                loss_cl[i], self.model.parameters(),
                retain_graph=True, create_graph=False
            )
            grad_vector = torch.cat([g.view(-1) for g in grad_i if g is not None])
            grad_norms.append(torch.norm(grad_vector, p=2))

        grad_norms = torch.stack(grad_norms)
        loss_grad = grad_norms.mean()

       
        loss = loss_cl_mean + loss_grad
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss_cl_mean, "train_grad_loss": loss_grad}

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        c = 0 
        for batch in tqdm(self.dataloaders["train"]):
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
            c+=1 
            if c>1000:
                break
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss


    # def perturb_weights(self, scale=0.0001):
    #     # Randomly perturb model parameters
    #     for param in self.model.parameters():
           
    #         param.data += torch.Tensor(np.random.uniform(low=(self.neighborhoodSize * -1) * self.P_T,
    #                                                     high=self.neighborhoodSize * self.P_T,
    #                                                     size=param.shape)).cuda()
            
    def perturb_weights(self, scale=0.0001):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                with torch.no_grad():
                    module.weight.data += torch.Tensor(
                        np.random.uniform(
                            low=(self.neighborhoodSize * -1) * self.P_T,
                            high=self.neighborhoodSize * self.P_T,
                            size=module.weight.shape
                        )
                    ).cuda()
                    
                    if module.bias is not None:
                        module.bias.data += torch.Tensor(
                            np.random.uniform(
                                low=(self.neighborhoodSize * -1) * self.P_T,
                                high=self.neighborhoodSize * self.P_T,
                                size=module.bias.shape
                            )
                        ).cuda()

    def save_best_weights(self):
        # Save the current best parameters
        self.best_params = [param.data.clone() for param in self.model.parameters()]

    def restore_best_weights(self):
        # Restore the best parameters
        with torch.no_grad():
            for param, best_param in zip(self.model.parameters(), self.best_params):
                param.data.copy_(best_param)

    def calculate_domain_gradients(self, x, y):
        """
        Compute gradients for each domain batch.
        """
        grads_i_v = []
        for i in range(x.shape[0]):
            logits, _ = self.model(x[i].unsqueeze(0))
            loss_i = F.cross_entropy(logits, y[i].unsqueeze(0))
            grad_i = autograd.grad(loss_i, self.model.parameters(), retain_graph=True)
            grads_i_v.append(torch.cat([g.flatten() for g in grad_i]))
        return grads_i_v

    def calculate_minimum_similarity(self, grads_i_v, targets, biases ):
        pairwise_combinations = [
            (i, j)
            for i, j in itertools.combinations(range(len(grads_i_v)), 2)
            if (targets[i] == targets[j] and biases[i] != biases[j])
        ]

        if not pairwise_combinations:
            return float('inf')  # or float('inf'), or raise an exception

        all_sims = [self.cos(grads_i_v[i], grads_i_v[j]) for i, j in pairwise_combinations]

        return min(all_sims)

    def get_grads(self):
        grads = []
        for p in self.model.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.model.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def update_simulated_annealing(self, x, y, b, **kwargs):
        """
        Search for better weights by adding noise and accepting them based on the improvement of
        the minimum domain pairwise gradient cosine similarity. The criterion can change.
        For i in range(iterations):
            # Step 1 --> Randomly add noise to weights to model parameters
            # Step 2 --> Calculate domain grads
            # Step 3 --> Calculate minimum pairwise cos similarity between domain grads and save
            # Step 4 --> Save model weights (state_dict()) if improvement
        # Step 5 --> load weights of optimal parameters
        # Step 6 --> update step
        """


        grads_i_v = self.calculate_domain_gradients(x, y)
        # cpu_grads = [g.detach().cpu().numpy() for g in grads_i_v]
        # cluster_labels = cluster_gradients(cpu_grads, eps=15.0, min_samples=2)
        # visualize_clusters(cpu_grads, cluster_labels, y.detach().cpu().numpy(), b.detach().cpu().numpy(), "tmp_init.png")

        best_min_sim = self.calculate_minimum_similarity(grads_i_v,y,b)
        if best_min_sim > 100:
            search_steps = 0
        else:
            search_steps = 20
        logits, _ = self.model(x)
        current_loss = F.cross_entropy(logits, y).item()

        # Save current state as best state before search
        current_params = [param.data.clone() for param in self.model.parameters()]
        self.save_best_weights()

        ##################
        # search_steps = 100
        best_search_loss = current_loss

        for search_step in range(search_steps):
            # Perturb weights
            self.perturb_weights()


            # Calculate domain grad sim in search
            grads_i_v = self.calculate_domain_gradients(x, y)

            step_min_sim = self.calculate_minimum_similarity(grads_i_v, y,b)
            
            logits, _ = self.model(x)
            step_loss = F.cross_entropy(logits, y)
            step_loss = step_loss.item()

            # Calculate change in loss
            delta_loss = step_loss - best_search_loss
            delta_sim = best_min_sim - step_min_sim
            relative_improvement = (step_min_sim - best_min_sim) / abs(best_min_sim + 1e-8)

            print(relative_improvement, best_min_sim, step_min_sim,delta_sim)
            # Accept or reject new weights
            if  relative_improvement >0.1:
                # cpu_grads = [g.detach().cpu().numpy() for g in grads_i_v]
                # cluster_labels = cluster_gradients(cpu_grads, eps=15.0, min_samples=2)
                # visualize_clusters(cpu_grads, cluster_labels, y.detach().cpu().numpy(), b.detach().cpu().numpy(), "tmp_updated.png")
                # Accept perturbed weights
                best_min_sim = step_min_sim
                best_search_loss = step_loss
                self.save_best_weights()
                with torch.no_grad():
                    for param, original_param in zip(self.model.parameters(), current_params):
                        param.data.copy_(original_param)
                break

        # Restore best weights (they may be original the params if search failed)
        self.restore_best_weights()
        logits, _ = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return {"loss": loss}
    
    def _method_specific_setups(self):
        self.cos = nn.CosineSimilarity(dim=0)

        self.neighborhoodSize = 0.005
        self.T = 1.0
        self.T_min = 0.01
        self.cooling_rate = 0.99
        self.P_T = 1.0
        self.P_T_min = 0.01
        self.perturb_cooling_rate = 0.99
        self.best_params = None
        self.best_loss = float('inf')