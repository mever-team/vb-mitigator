import os
import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc
import torch.nn.functional as F
from tools.utils import (
    log_msg,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    setup_logger,
)
from tools.task_vectors import TaskVector


class ModelEditingTrainer(BaseTrainer):

    def _setup_models(self):
        super()._setup_models()

        self.bcc_nets = get_bcc(self.cfg, self.num_class)

        for _, bcc_net in self.bcc_nets.items():
            checkpoint = load_checkpoint("/home/isarridis/projects/vb-mitigator/output/biased_mnist_baselines/corr1/erm/latest")
            bcc_net.load_state_dict(checkpoint["model"])
            print(f"Loaded checkpoint from ./output/biased_mnist_baselines/corr1/erm/latest")
            bcc_net.to(self.device)
            bcc_net.eval()
            
    def eval(self):
        self.load_checkpoint("best")
        for _, bcc_net in self.bcc_nets.items():
            self.bcc_net = bcc_net
        task_vector = TaskVector(self.model, self.bcc_net)
        # Negate the task vector
        neg_task_vector = -task_vector
        # Apply the task vector
        self.model = neg_task_vector.apply_to(self.model, scaling_coef=1)
    
        test_performance = self._validate_epoch(stage="test")
        test_log_dict = self.build_log_dict(test_performance, stage="test")
        print(f"Test performance: {test_log_dict}")

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        
        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss