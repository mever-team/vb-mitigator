import math
import os
from tqdm import tqdm
from my_datasets.utk_face import get_utk_face
from tools.utils import load_checkpoint
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
from .losses import (
    ContrastiveLoss,
    DistillKL,
    FocalLoss,
    LDAMLoss,
    MAELoss,
    NCELoss,
    NCEandAGCE,
    NCEandAUE,
    RCELoss,
    SubcenterArcMarginProduct,
    LabelSmoothSoftmaxCEV1,
    ArcMarginProduct,
    WBLoss,
)
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
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
import torchvision

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads


# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        _, features = self.backbone(x)
        z = self.projection_head(features)
        return z

    def predict(self, x):
        _, features = self.backbone(x)
        z = self.projection_head(features)
        z = self.fc(z)
        return z


class LowCorrelationPairDataset(Dataset):
    def __init__(self, base_dataset, pair_indices_path):
        """
        Args:
            base_dataset: a dataset where __getitem__(i) returns a dict with 'inputs' and 'targets'
            pair_indices_path: path to the saved .npy file containing (N, 2) index pairs
        """
        self.dataset = base_dataset
        self.pairs = np.load(pair_indices_path)
        print(len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        item_i = self.dataset[i]
        item_j = self.dataset[j]

        # Assuming each item is a dict with 'inputs' and 'targets'
        return {
            "x_i": item_i["inputs"],
            "y_i": item_i["targets"],
            "x_j": item_j["inputs"],
            "y_j": item_j["targets"],
            "b_i": item_i["race"],
            "b_j": item_j["race"],
        }


def cluster_by_class(features, targets, n_clusters=2):
    clusters = {}
    for cls in torch.unique(targets):
        cls_idx = (targets == cls).nonzero(as_tuple=True)[0]
        feats_cls = features[cls_idx]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feats_cls)
        clusters[int(cls.item())] = {
            "indices": cls_idx,
            "labels": torch.tensor(kmeans.labels_),
        }
    return clusters


def estimate_cluster_accuracy(targets, predictions, clusters):
    cluster_accuracy = {}
    for cls, cluster_data in clusters.items():
        idxs = cluster_data["indices"]
        labels = cluster_data["labels"]
        accs = []
        for label in [0, 1]:
            group_idx = idxs[labels == label]
            correct = 0
            total = len(group_idx)
            for i in group_idx:
                y = targets[i]
                pred = predictions[i]
                correct += int(pred == y)
            acc = correct / total if total > 0 else 0
            accs.append(acc)
        cluster_accuracy[cls] = accs
    print(cluster_accuracy)
    return cluster_accuracy


def get_sample_weights(clusters, cluster_acc, targets, indices, l=0.66):
    weights = {}
    for cls, accs in cluster_acc.items():
        hard_cluster = int(np.argmin(accs))
        cls_indices = clusters[cls]["indices"]
        cluster_labels = clusters[cls]["labels"]
        for i, label in zip(cls_indices, cluster_labels):
            idx = indices[i].item()
            if label == hard_cluster:
                weights[idx] = 0.33
            else:
                weights[idx] = 0.33 + l
    return weights


class ConTrainer(BaseTrainer):
    def _setup_resume(self):
        return

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.model = SimCLR(self.model)
        self.model.to(self.device)

    def _setup_optimizer(self):
        super()._setup_optimizer()

        self.optimizer_fc = torch.optim.Adam(
            self.model.fc.parameters(),
            lr=0.001,
            weight_decay=1e-4,
        )

    def _setup_criterion(self):
        # self.criterion = LabelSmoothSoftmaxCEV1()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_con = ContrastiveLoss(margin=2.0)
        # self.criterion_kd = DistillKL(T=4)
        self.criterion_ssl = loss.NTXentLoss(temperature=0.5)

    def _train_iter(self, batch):
        inputs = batch["inputs"]
        # print(inputs[0].shape)
        # images = torch.stack([sample for sample in inputs], dim=0)
        view1, view2 = inputs[0], inputs[1]
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)
        # total_images = torch.cat([inputs[:,0], inputs[:,1]], dim=0)
        # total_images = total_images.to(self.device)

        indices = batch["index"]
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)

        # if self.current_epoch <= 10:
        #     targets = batch["targets"].to(self.device)
        # else:
        #     targets = torch.tensor(
        #         [self.preds[int(i)] for i in indices],
        #         dtype=torch.long,
        #         device=self.device
        #     )

        self.optimizer.zero_grad()
        # total_features = self.model(total_images)
        outputs1 = self.model(view1)
        outputs2 = self.model(view2)
        # outputs1, outputs2 = torch.split(
        #     total_features,
        #     [int(self.cfg.SOLVER.BATCH_SIZE / 2), int(self.cfg.SOLVER.BATCH_SIZE / 2)],
        #     dim=0,
        # )
        # outputs = torch.stack([
        #     torch.max(outputs[:, 0:10], dim=1).values,   # max over logits 0–10
        #     torch.max(outputs[:, 11:20], dim=1).values   # max over logits 11–20
        # ], dim=1)

        # Convert to probabilities
        # outputs = F.softmax(outputs, dim=1)

        # N = targets.shape[0]
        # weight_matrix = torch.zeros(N, 2, device=weights.device)

        # # Fill appropriate class column with the weight
        # weight_matrix[torch.arange(N), targets] = weights
        # print(outputs)
        # print(weights)
        # outputs += (weight_matrix.detach()  ) * 10
        # print(weights)
        # Apply sample weights to loss
        # margins = self.margins[indices].to(outputs.dtype).to(self.device)

        # margins = torch.zeros_like(targets, dtype=torch.float32).to(self.device) + 0.1
        # mask = targets != biases
        # margins[mask]+= 0.8
        # if self.current_epoch>1:
        #     weights = torch.tensor(
        #     [self.weights[int(i)] for i in indices],
        #     dtype=torch.float32,
        #     device=self.device
        #     )
        # else :
        #     weights = torch.zeros_like(targets, dtype=torch.float32).to(self.device) + 1
        # loss = F.cross_entropy(outputs,targets,reduction="none") * weights
        # loss = loss.mean()
        loss = self.criterion_ssl(outputs1, outputs2)
        # loss = (loss * weights).mean()
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}

    def _train_iter_fc(self, batch):
        inputs = batch["inputs"]
        inputs = inputs.to(self.device)
        indices = batch["index"]
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)

        self.optimizer_fc.zero_grad()
        outputs = self.model.predict(inputs)

        loss = F.cross_entropy(outputs, targets)
        self._loss_backward(loss)
        self.optimizer_fc.step()
        return {"train_cls_loss": loss}

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        outputs = self.model.predict(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss

    def freeze(self, flag):
        for param in self.model.parameters():
            param.requires_grad = not flag
        for param in self.model.fc.parameters():
            param.requires_grad = flag

    def _train_epoch(self):
        self._set_train()
        # self.teacher.eval()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        self.freeze(False)
        for batch in tqdm(self.dataloaders["train_con"]):
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch)
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        self.freeze(True)
        for batch in tqdm(self.dataloaders["train"]):
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter_fc(batch)
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}

        if self.current_epoch == 199:  # or self.current_epoch == 8:
            self.collect_all()
            self.clustering()
            # self._setup_models()
            # self._setup_optimizer()
            # self._setup_scheduler()
        return avg_loss

    def _method_specific_setups(self):
        transform = transforms.SimCLRTransform(input_size=64, cj_prob=0.5)

        self.dataloaders["train_con"], _ = get_utk_face(
            self.cfg.DATASET.UTKFACE.ROOT,
            batch_size=512,
            split="train",
            bias_attr=self.cfg.DATASET.UTKFACE.BIAS,
            ratio=self.cfg.DATASET.UTKFACE.RATIO,
            transform=transform,
            two_crop=False,
        )

    def collect_preds(self):
        self.model.eval()
        self.preds = {}
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
                preds = torch.argmax(outputs, dim=1)
                # logits = self.metric_loss.predict(features,targets)

                # self.losses = {**self.losses, **{idx.item() : loss[i].item() for idx, i in zip(indices,range(targets.shape[0]))} }
                for idx, pred in zip(indices, preds):
                    idx_item = idx.item()
                    self.preds[idx_item] = pred.item()
        self.model.train()

    def collect_all(self):
        self.model.eval()
        self.preds = []
        self.targets = []
        self.features = []
        self.indices = []
        with torch.no_grad():
            for batch in tqdm(self.dataloaders["train"]):
                bsz = batch["targets"].shape[0]
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                indices = batch["index"]
                biases = batch[self.biases[0]]
                features = self.model(inputs)
                # if not isinstance(outputs, tuple):
                #     raise ValueError("Model output must be a tuple (logits, features)")
                # outputs, features = outputs
                # preds = torch.argmax(outputs,dim=1)
                # logits = self.metric_loss.predict(features,targets)
                self.preds.append(biases.cpu())
                self.targets.append(targets.cpu())
                self.features.append(features.cpu())
                self.indices.append(torch.tensor(indices))

        # Concatenate everything
        self.preds = torch.cat(self.preds)
        self.targets = torch.cat(self.targets)
        self.features = torch.cat(self.features)
        self.indices = torch.cat(self.indices)
        self.model.train()

        feats = self.features.numpy()
        biases = self.preds.numpy()  # (N,)
        targets = self.targets.numpy()  # (N, T)
        indices_all = self.indices.numpy()
        save_path = "timeseries_data_feats.npz"
        np.savez(
            save_path,
            biases=biases,
            targets=targets,
            feats=feats,
            indices_all=indices_all,
        )

    def clustering(self):
        self.clusters = cluster_by_class(self.features, self.targets)
        accs = estimate_cluster_accuracy(self.targets, self.preds, self.clusters)
        self.weights = get_sample_weights(
            self.clusters, accs, self.targets, self.indices
        )
