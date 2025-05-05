"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from Gradient Starvation:
# https://github.com/mohammadpz/Gradient_Starvation
# --------------------------------------------------------

from tools.metrics.utils import AverageMeter
import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer
from models.builder import get_model
import torch.nn as nn

class MultiHeadNet(nn.Module):
    def __init__(self, main, num_classes, **kwargs):
        super(MultiHeadNet, self).__init__()
        self.main = main
        self.proj = nn.Linear(self.main.fc.in_features,self.main.fc.in_features)
        self.fc = nn.Linear(self.main.fc.in_features,num_classes)

    def forward(self, x):
        logits_main, features = self.main(x)
        features_detached = features.detach()
        logits_aux = self.fc(self.proj(features_detached))
        return [logits_main, logits_aux], features


class MultiHeadTrainer(BaseTrainer):

    def _setup_models(self):
        self.main = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.main.parameters():
                param.requires_grad = False
            for param in self.main.fc.parameters():
                param.requires_grad = True

        self.model = MultiHeadNet(self.main, self.num_class)
        self.model.to(self.device)


    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        logits, features = self.model(inputs)

        # MAIN OPTIMIZER STEP
        self.optimizer.zero_grad()
        loss_main = self.criterion(logits[0] + logits[1].detach(), targets)
        loss_main.backward(retain_graph=True)
        self.optimizer.step()

        # AUX OPTIMIZER STEP
        self.optimizer_aux.zero_grad()
        loss_aux = self.criterion(logits[1], targets)
        loss_aux.backward()
        self.optimizer_aux.step()

        return {"train_loss_main": loss_main, "train_loss_aux": loss_aux}
    

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
        self.scheduler_aux.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss
    
    def _setup_optimizer(self):
        main_params = list(self.model.main.parameters())
        aux_params = list(self.model.proj.parameters()) + list(self.model.fc.parameters())

        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                main_params,
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
            self.optimizer_aux = torch.optim.SGD(
                aux_params,
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE == "Adam":
            self.optimizer = torch.optim.Adam(
                main_params,
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
            self.optimizer_aux = torch.optim.Adam(
                aux_params,
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")


    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        outputs = outputs[0]
        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss
    


    def _setup_scheduler(self):
        if self.cfg.SOLVER.SCHEDULER.TYPE == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.SOLVER.SCHEDULER.STEP_SIZE,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
        elif self.cfg.SOLVER.SCHEDULER.TYPE == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
            self.scheduler_aux = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_aux,
                milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
        elif self.cfg.SOLVER.SCHEDULER.TYPE == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.SOLVER.SCHEDULER.T_MAX
            )
        elif self.cfg.SOLVER.SCHEDULER.TYPE == "None":
            return
        else:
            raise ValueError(
                f"Unsupported scheduler type: {self.cfg.SOLVER.SCHEDULER.TYPE}"
            )