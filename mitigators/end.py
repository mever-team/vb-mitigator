from .losses import EnDLoss
from .base_trainer import BaseTrainer


class EndTrainer(BaseTrainer):

    def _setup_criterion(self):
        super()._setup_criterion()
        self.criterion_train = EnDLoss(
            self.cfg.MITIGATOR.END.ALPHA, self.cfg.MITIGATOR.END.BETA
        )

    def _method_specific_setups(self):
        if len(self.biases) > 1:
            raise ValueError(
                "Multiple biases detected! END can be applied only to single attribute biases. Please select another dataset."
            )

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        biases = batch[self.biases[0]].to(self.device)

        self.optimizer.zero_grad()
        outputs, feats = self.model(inputs)

        ce_loss, end_loss = self.criterion_train(outputs, targets, biases, feats)

        loss = ce_loss + self.cfg.MITIGATOR.END.WEIGHT * end_loss

        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": ce_loss, "train_end_loss": end_loss}
