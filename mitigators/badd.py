import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc


class BAddTrainer(BaseTrainer):

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.model.to(self.device)

        self.bcc_nets = get_bcc(self.cfg, self.num_class)

        for _, bcc_net in self.bcc_nets.items():
            bcc_net.to(self.device)
            bcc_net.eval()

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()

        pr_feats = []
        for _, bcc_net in self.bcc_nets.items():
            with torch.no_grad():
                _, pr_feat = bcc_net(inputs)
                pr_feats.append(pr_feat)

        outputs = self.model.badd_forward(inputs, pr_feats, self.cfg.MITIGATOR.BADD.M)

        loss_cl = self.criterion(outputs, targets)
        loss = loss_cl
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss_cl}
