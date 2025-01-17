import torch
from models.builder import get_model, get_bcc
from .base_trainer import BaseTrainer


class BAddTrainer(BaseTrainer):
    """
    BAddTrainer is a specialized trainer class that extends the BaseTrainer.
    It is responsible for setting up models and training iterations for a specific
    type of model that uses BCC networks and a BADD forward pass.

    Methods
    -------
    _setup_models():
        Initializes and sets up the main model and BCC networks, moving them to the
        appropriate device (e.g., GPU).

    _train_iter(batch):
        Performs a single training iteration, including forward pass, loss computation,
        and backpropagation. Returns a dictionary containing the classification loss.

    Attributes
    ----------
    model : torch.nn.Module
        The main model used for training.
    bcc_nets : dict
        A dictionary of BCC networks used to extract features.
    device : torch.device
        The device (CPU or GPU) on which the models are run.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    criterion : callable
        The loss function used to compute the training loss.
    cfg : object
        Configuration object containing model and training settings.
    num_class : int
        The number of classes in the classification task.
    """

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, self.cfg.MODEL.PRETRAINED
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
