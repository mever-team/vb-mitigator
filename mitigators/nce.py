import math
import os
from tqdm import tqdm
from tools.utils import load_checkpoint
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
from .losses import NCELoss, NCEandAGCE, NCEandAUE, SubcenterArcMarginProduct, LabelSmoothSoftmaxCEV1, ArcMarginProduct, WBLoss
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
import copy


class NCETrainer(BaseTrainer):

    
    def _method_specific_setups(self):
        self.nce_model = copy.deepcopy(self.model)
        checkpoint = load_checkpoint("/home/isarridis/projects/vb-mitigator/output/utkface_baselines/race/lr/latest")
        self.nce_model.load_state_dict(checkpoint["model"])

        self.ce_model = copy.deepcopy(self.model)
        checkpoint = load_checkpoint("/home/isarridis/projects/vb-mitigator/output/utkface_baselines/race/erm/latest")
        self.ce_model.load_state_dict(checkpoint["model"])
        self.compute_predictions()
        self.model.load_state_dict(checkpoint["model"])
        # print(sdg)


    def compute_predictions(self):
        self.nce_model.eval()
        self.ce_model.eval()

        targets_all = []
        biases_all = []
        indices_all = []
        nce_logits_all = []
        nce_preds_all = []
        nce_features_all = []

        ce_logits_all = []
        ce_preds_all = []
        ce_features_all = []

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["train"]):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                biases = batch[self.biases[0]].to(self.device)
                indices = batch["index"]

                # NCE model forward
                logits_nce, features_nce = self.nce_model(inputs)
                preds_nce = torch.argmax(logits_nce, dim=1)

                # CE model forward
                logits_ce, features_ce = self.ce_model(inputs)
                preds_ce = torch.argmax(logits_ce, dim=1)

                # Append outputs
                targets_all.append(targets.cpu())
                biases_all.append(biases.cpu())
                indices_all.append(indices.cpu())

                nce_logits_all.append(logits_nce.cpu())
                nce_preds_all.append(preds_nce.cpu())
                nce_features_all.append(features_nce.cpu())

                ce_logits_all.append(logits_ce.cpu())
                ce_preds_all.append(preds_ce.cpu())
                ce_features_all.append(features_ce.cpu())

        # Stack and convert to numpy
        np.savez(
            "nce_ce_outputs.npz",
            targets=torch.cat(targets_all).numpy(),
            biases=torch.cat(biases_all).numpy(),
            indices=torch.cat(indices_all).numpy(),

            nce_logits=torch.cat(nce_logits_all).numpy(),
            nce_preds=torch.cat(nce_preds_all).numpy(),
            nce_features=torch.cat(nce_features_all).numpy(),

            ce_logits=torch.cat(ce_logits_all).numpy(),
            ce_preds=torch.cat(ce_preds_all).numpy(),
            ce_features=torch.cat(ce_features_all).numpy(),
        )

        print("Saved outputs to 'nce_ce_outputs.npz'")

        all_preds = torch.cat(nce_preds_all)
        all_targets = torch.cat(targets_all)
        all_indices = torch.cat(indices_all)

        misclassified_mask = all_preds != all_targets
        misclassified_indices = all_indices[misclassified_mask].tolist()

        print(f"Total misclassified samples by NCE model: {len(misclassified_indices)}")

        # Now create a filtered dataset and dataloader
        from torch.utils.data import Subset, DataLoader

        misclassified_subset = Subset(self.dataloaders["train"].dataset, misclassified_indices)
        self.dataloaders["train"] = DataLoader(
            misclassified_subset,
            batch_size=self.dataloaders["train"].batch_size,
            shuffle=False,
            num_workers=self.dataloaders["train"].num_workers,
            pin_memory=True
        )

               