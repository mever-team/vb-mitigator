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

def pairwise_distances(a, b=None, eps=1e-6):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a**2, dim=1)
    bb = torch.sum(b**2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)
    dists = torch.sqrt(dists + eps)
    return dists

def find_zscore_outliers(features, threshold=3):
    mean = torch.mean(features, dim=0)
    std = torch.std(features, dim=0)
    z_scores = torch.abs((features - mean) / (std + 1e-8)) # Add epsilon for stability
    outlier_mask = torch.any(z_scores > threshold, dim=1)
    outlier_indices = torch.where(outlier_mask)[0]
    return outlier_indices
def compute_tp_rate(predictions, ground_truth):
    """
    Computes the True Positive Rate (TPR) or Sensitivity for two boolean tensors.

    Args:
        predictions (torch.BoolTensor): The predicted boolean values.
        ground_truth (torch.BoolTensor): The true boolean values.

    Returns:
        torch.Tensor: The True Positive Rate (TPR) as a scalar tensor.
                      Returns 0 if there are no actual positive cases.
    """
    # if not isinstance(predictions, torch.BoolTensor) or not isinstance(ground_truth, torch.BoolTensor):
    #     raise TypeError("Both predictions and ground_truth must be torch.BoolTensor.")

    if predictions.shape != ground_truth.shape:
        raise ValueError("Predictions and ground_truth tensors must have the same shape.")

    # True Positives (TP): predictions are True AND ground_truth are True
    tp = torch.sum(predictions & ground_truth).float()

    # Actual Positives (P): where ground_truth is True
    p = torch.sum(ground_truth).float()

    # True Positive Rate (TPR) = TP / P
    if p > 0:
        tpr = tp / p
    else:
        tpr = torch.tensor(1.0)
        
    pp = torch.sum(predictions).float()

    # True Positive Rate (TPR) = TP / P
    if pp > 0:
        prec = tp / pp
    else:
        prec = torch.tensor(1.0)
    return tpr, prec
def var_loss(loss, features, labels, biases):
    loss= loss.detach()
    features = F.normalize(features, dim=1)
    th = torch.mean(loss) + 2.5*torch.std(loss) 
    
    # bc_idcs = find_zscore_outliers(features)
    # bc_samples = bc_idcs.shape[0]
    features_ba = features[loss<th]
    # features_ba = features[labels==biases]
    # print(torch.std(loss), torch.sum(biases != labels))
    
    # print(loss)

    # print(torch.sum(biases != labels), torch.sum(th < loss))

    # print(torch.sum(torch.min(loss[biases != labels]) > torch.max(loss[biases == labels])))
    # print(torch.sum(biases != labels) == torch.std(loss) < 0.25)
    if torch.std(loss) < 0.25:
        return torch.tensor(0.0).to(labels.device)
    
    v_loss = torch.tensor(0.0).to(labels.device)
    
    for i in range(2):
        loss_i = loss[labels==i]
        features_i = features[labels==i]
        
        labels_i = labels[labels==i]
        biases_i = biases[labels==i]

        # mask1 = labels_i == biases_i
        # mask2 = labels_i != biases_i
        mask1 = loss_i <= th
        mask2 = loss_i > th

        mask1 = mask1.to(labels_i.device)
        mask2 = mask2.to(labels_i.device)

        # if mask is empty, return zero
        if sum(mask1) == 0:
            continue
        
        if sum(mask2) == 0:
            continue

        # print(compute_tp_rate(mask2,labels_i != biases_i))
        features_i_ba = features_i[mask1]
        features_i_ba_mean = torch.mean(features_i_ba,dim=0)
        features_i_bc = features_i[mask2]
        features_i_bc_mean = torch.mean(features_i_bc,dim=0)

        # Ensure mean vectors have the correct shape for broadcasting
        features_i_ba_mean = features_i_ba_mean.unsqueeze(0)  # (1, feature_dim)
        features_i_bc_mean = features_i_bc_mean.unsqueeze(0)  # (1, feature_dim)

        # Calculate cosine similarity
        # similarities_ba = F.cosine_similarity(features_ba, features_i_ba_mean, dim=1)
        # similarities_bc = F.cosine_similarity(features_ba.detach(), (features_i_bc_mean * features_i_ba_mean.detach())/2, dim=1)

        similarities_ba = pairwise_distances(features_ba, features_i_ba_mean)
        similarities_ba = 1.0 / (1 + similarities_ba**1)

        similarities_bc = pairwise_distances(features_ba.detach(), features_i_bc_mean)
        similarities_bc = 1.0 / (1 + similarities_bc**1)
        # print(similarities_ba[labels_i==biases_i])
        # print(similarities_bc)
        # convert to probabilities
        similarities_ba = similarities_ba / (
            torch.sum(similarities_ba)
        )
        similarities_ba = similarities_ba #[mask1]
        similarities_bc = similarities_bc #[mask1]
        similarities_bc = similarities_bc / (
            torch.sum(similarities_bc)
        )

        # Jeffrey's divergence
        l = (similarities_ba.detach() - similarities_bc) * (
            torch.log(similarities_ba.detach()) - torch.log(similarities_bc)
        )
        v_loss += torch.mean(l)
        # print(l)
        # print(similarities_bc,similarities_ba)
    # print( torch.sum(biases != labels), torch.sum(th < loss))
    return v_loss




class GCosTrainer(BaseTrainer):

    def _setup_resume(self):
        self.losses = {}
        return

    def _train_iter(self, batch):
        
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        indices = batch["index"]
        biases = batch[self.biases[0]].to(self.device)

        
        # if self.current_epoch>0:
            # grads = get_per_sample_gradients(self.model, inputs, targets, self.criterion)
            # # grads = [g/np.linalg.norm(g) for g in grads]
            # detect_gradient_outliers(grads, targets.detach().cpu().numpy(),biases.detach().cpu().numpy())

            # cluster_labels = cluster_gradients(grads, eps=15.0, min_samples=2)
            # visualize_clusters(grads, cluster_labels, targets.detach().cpu().numpy(), biases.detach().cpu().numpy())
        #     raise ValueError()

        # outputs = self.model(inputs)
        # if not isinstance(outputs, tuple):
        #     raise ValueError("Model output must be a tuple (logits, features)")
        # outputs, features = outputs


        # loss = self.criterion(outputs, targets)

       
        
        # self._loss_backward(loss)

        # # Step 4: average projected gradients and manually assign to model parameters
        # grads = get_per_sample_gradients(self.model, inputs, targets, self.criterion)
        # outliers_per_class = detect_gradient_outliers(grads, targets.detach().cpu().numpy(),biases.detach().cpu().numpy())

        # # Step 3: Build class-wise gradient vectors
        # grads = np.array(grads)
        # true_labels = targets.detach().cpu().numpy()

        # updated_grads = []

        # for cls in np.unique(true_labels):
        #     idx_cls = np.where(true_labels == cls)[0]
        #     idx_outliers = outliers_per_class[cls]
        #     idx_core = [i for i in idx_cls if i not in idx_outliers]

        #     if len(idx_core) > 0:
        #         avg_grad_core = np.mean(grads[idx_core], axis=0)
        #     else:
        #         avg_grad_core = np.zeros_like(grads[0])  # fallback if no core samples
            
        #     # For each outlier, add its gradient to the core gradient
        #     if len(idx_outliers) > 0:
        #         updated_grads.append(avg_grad_core)
        #         for i in idx_outliers:
        #             updated_grads.append(grads[i])
        # if len(updated_grads) == 0:
        #     return {"train_cls_loss": torch.tensor(0.0)}  
        # # Step 4: Average across all classes
        # final_grad = np.mean(updated_grads, axis=0)
        # final_grad = torch.tensor(final_grad, dtype=torch.float32, device=self.device)

        # # Step 5: Apply to model parameters
        # offset = 0
        # self.model.zero_grad()
        # for param in self.model.parameters():
        #     if param.requires_grad:
        #         num_param = param.numel()
        #         grad_slice = final_grad[offset:offset + num_param].view_as(param)
        #         param.grad = grad_slice
        #         offset += num_param

        # ba_mask = targets == biases 
        # bc_mask = ~ba_mask

    
        # if not bc_mask.any():  # Skip update if no bias-conflicting samples
        #     return {"train_cls_loss": torch.tensor(0.0)}

        # # Compute losses separately
        # loss_ba = self.criterion(outputs[ba_mask], targets[ba_mask]) if ba_mask.any() else 0.0
        # loss_bc = self.criterion(outputs[bc_mask], targets[bc_mask])

        # loss = loss_bc + loss_ba

        # self.optimizer.zero_grad()
        # self._loss_backward(loss)
        # self._optimizer_step()
        if self.current_epoch < 4:
            outputs = self.model(inputs)
            if not isinstance(outputs, tuple):
                raise ValueError("Model output must be a tuple (logits, features)")
            outputs, features = outputs


            loss = self.criterion(outputs, targets)
            # self.losses = {**self.losses, **{idx.item() : loss[i].item() for idx, i in zip(indices,range(targets.shape[0]))} }
            # loss = loss.mean()
            self.optimizer.zero_grad()
            self._loss_backward(loss)
            self._optimizer_step()
            return  {"loss": loss, "loss_cl": loss, "loss_var": loss} 
        else:
            out = self.update_simulated_annealing(inputs,targets,biases, indices)

        # raise ValueError()
            return out

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
            if c>10000:
                break
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        if self.current_epoch == 1:
            self.collect_losses()
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

    def collect_losses(self): 
        self.model.eval()
        self.losses = {}
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


                loss = F.cross_entropy(outputs, targets, reduction="none")
                self.losses = {**self.losses, **{idx.item() : loss[i].item() for idx, i in zip(indices,range(targets.shape[0]))} }
        # self._setup_models()
        # self._setup_optimizer()
        # self._setup_scheduler()
        self.model.train()

    def update_simulated_annealing(self, x, y, b, indices, **kwargs):
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
        logits, features = self.model(x)
        loss_cl = self.criterion(logits, y)
        # print(len(self.losses))
        # for i in indices:
        #     print(self.losses[i.item()])
        loss_pre = torch.tensor([self.losses[idx.item()] for idx in indices])
        loss_v = 1000*var_loss(loss_pre.to(self.device), features, y, b)
        loss =  loss_v + loss_cl
        # if loss_v > 0:
        # loss_cl = loss_cl.mean() 
            
        self.optimizer.zero_grad()
        loss.backward()
        # print({"loss": loss, "loss_cl": loss_cl, "loss_var": loss_v} )
        self.optimizer.step()
        return  {"loss": loss, "loss_cl": loss_cl, "loss_var": loss_v} 

        # grads_i_v = self.calculate_domain_gradients(x, y)
        # cpu_grads = [g.detach().cpu().numpy() for g in grads_i_v]
        # cluster_labels = cluster_gradients(cpu_grads, eps=15.0, min_samples=2)
        # visualize_clusters(cpu_grads, cluster_labels, y.detach().cpu().numpy(), b.detach().cpu().numpy(), "tmp_init.png")

        # best_min_sim = self.calculate_minimum_similarity(grads_i_v,y,b)
        # mask = y == b
        # if torch.sum(~mask) == 0:
        #     logits, features = self.model(x)
        #     loss_cl = F.cross_entropy(logits, y)
        #     loss_v = var_loss(loss, features, y)
        #     loss = loss_cl + loss_v
        #     self.optimizer.zero_grad()
        #     loss.backward()

        #     self.optimizer.step()
        #     return  {"loss": loss} 
        # search_steps = 250
        # # if best_min_sim > 100:
        # #     search_steps = 0
        # # if best_min_sim < -0.14:
        # #     search_steps = 250
        # # else: 
        # #     search_steps = 0
        # # print(best_min_sim)
        # logits, _ = self.model(x)
        # loss = F.cross_entropy(logits, y)
        # # loss, ba_loss, bc_loss = self.variance_penalized_ce_loss(logits,y,1)
        # self.optimizer.zero_grad()
        # loss.backward()

        # self.optimizer.step()

        # # ba_predictions = torch.argmax(logits[mask], dim=1)
        # # ba_accuracy = (ba_predictions == y[mask]).float().mean().item()
        # # bc_predictions = torch.argmax(logits[~mask], dim=1)
        # # bc_accuracy = (bc_predictions == y[~mask]).float().mean().item()

        # # print(f"{ba_loss:.2f}, {bc_loss:.2f}, {ba_accuracy:.2f}, {bc_accuracy:.2f}")

        # # with torch.no_grad():
        # #     logits, _ = self.model(x)

        # # ba_loss = F.cross_entropy(logits[mask], y[mask])
        # # bc_loss = F.cross_entropy(logits[~mask], y[~mask])
        # # best_min_sim = torch.abs(ba_loss-bc_loss).item()

        # # current_loss = F.cross_entropy(logits, y).item()

        # # # Save current state as best state before search
        # # current_params = [param.data.clone() for param in self.model.parameters()]
        # # self.save_best_weights()

        # # ##################
        # # # search_steps = 100
        # # best_search_loss = current_loss

        # # for search_step in range(search_steps):
        # #     # Perturb weights
        # #     self.perturb_weights()


        # #     # Calculate domain grad sim in search
        # #     # grads_i_v = self.calculate_domain_gradients(x, y)

        # #     # step_min_sim = self.calculate_minimum_similarity(grads_i_v, y,b)
            
        # #     logits, _ = self.model(x)

        # #     ba_loss = F.cross_entropy(logits[mask], y[mask])
        # #     bc_loss = F.cross_entropy(logits[~mask], y[~mask])

        # #     ba_predictions = torch.argmax(logits[mask], dim=1)
        # #     ba_accuracy = (ba_predictions == y[mask]).float().mean().item()
        # #     bc_predictions = torch.argmax(logits[~mask], dim=1)
        # #     bc_accuracy = (bc_predictions == y[~mask]).float().mean().item()
                 

        # #     step_min_sim = torch.abs(ba_loss-bc_loss).item()

        # #     step_loss = F.cross_entropy(logits, y)
        # #     step_loss = step_loss.item()

        # #     # Calculate change in loss
        # #     delta_loss = step_loss - best_search_loss
        # #     delta_sim = step_min_sim - best_min_sim

        # #     # print(delta_sim, step_min_sim)
        # #     # Accept or reject new weights
        # #     # if  relative_improvement >0.1:
        # #     if delta_sim <0 :
        # #         print(f"{delta_sim:.2f}, {step_min_sim:.2f}, {ba_loss.item():.2f}, {bc_loss.item():.2f}, {ba_accuracy:.2f}, {bc_accuracy:.2f}")
        # #         # cpu_grads = [g.detach().cpu().numpy() for g in grads_i_v]
        # #         # cluster_labels = cluster_gradients(cpu_grads, eps=15.0, min_samples=2)
        # #         # visualize_clusters(cpu_grads, cluster_labels, y.detach().cpu().numpy(), b.detach().cpu().numpy(), "tmp_updated.png")
        # #         # Accept perturbed weights
        # #         best_min_sim = step_min_sim
        # #         best_search_loss = step_loss
        # #         self.save_best_weights()
        # #         with torch.no_grad():
        # #             for param, original_param in zip(self.model.parameters(), current_params):
        # #                 param.data.copy_(original_param)
        # #         # break

        # # # Restore best weights (they may be original the params if search failed)
        # # self.restore_best_weights()
        # # # logits, _ = self.model(x)
        # # # loss = F.cross_entropy(logits, y)
        # # # self.optimizer.zero_grad()
        # # # loss.backward()

        # # # self.optimizer.step()

        # return {"loss": loss}
    
    def _method_specific_setups(self):
        self.cos = nn.CosineSimilarity(dim=0)

        self.neighborhoodSize = 0.03
        self.T = 1.0
        self.T_min = 0.01
        self.cooling_rate = 0.99
        self.P_T = 1.0
        self.P_T_min = 0.01
        self.perturb_cooling_rate = 0.99
        self.best_params = None
        self.best_loss = float('inf')

    def variance_penalized_ce_loss(self, logits, targets, variance_penalty_factor=1000):
        loss_per_sample = F.cross_entropy(logits, targets, reduction='none')
        mean_loss = torch.mean(loss_per_sample)
        variance_loss = torch.var(loss_per_sample)
        total_loss = mean_loss + variance_penalty_factor * variance_loss
        return total_loss, mean_loss.item(), variance_loss.item()