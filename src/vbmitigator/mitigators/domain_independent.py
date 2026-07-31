
from vbmitigator.models.domain_independent_classifier import DomainIndependentClassifier

from .base_trainer import BaseTrainer


class DomainIndependentTrainer(BaseTrainer):

    def _setup_models(self):
        self.model = DomainIndependentClassifier(
            self.cfg.MODEL.TYPE,
            self.num_class,
            self.num_biases,
            self.cfg.MODEL.PRETRAINED,
        ).to(self.device)

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        if len(self.biases) == 1:
            domain_label = batch[self.biases[0]].to(self.device)
        else:
            domain_label = targets * 0
            for i, bias in enumerate(self.biases):
                domain_label += batch[bias].to(self.device) * (i + 1)
            domain_label = domain_label.to(self.device)

        self.optimizer.zero_grad()
        logits_per_domain = self.model(inputs)
        logits = logits_per_domain[range(logits_per_domain.shape[0]), domain_label]

        loss = self.criterion(logits, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss}
