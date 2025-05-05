import math
import os
from tqdm import tqdm
from datasets.utk_face import get_utk_face
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
from .losses import ContrastiveLoss, DistillKL, FocalLoss, LDAMLoss, MAELoss, NCELoss, NCEandAGCE, NCEandAUE, RCELoss, SubcenterArcMarginProduct, LabelSmoothSoftmaxCEV1, ArcMarginProduct, WBLoss
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

class DinoTrainer(BaseTrainer):
    def _setup_resume(self):
        return
    
    def _setup_models(self):
        self.model = nn.Linear(768, self.num_class) 
        self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.to(self.device)
        self.extractor.to(self.device)
        self.extractor.eval()

    
      

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        with torch.no_grad():
            inputs = self.extractor.forward_features(inputs)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}
   
    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        with torch.no_grad():
            inputs = self.extractor.forward_features(inputs)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss
    
   
