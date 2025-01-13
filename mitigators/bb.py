import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from .losses import BBLoss
from .base_trainer import BaseTrainer


class BBTrainer(BaseTrainer):

    # def _setup_criterion(self):
    #     # Example inputs
    #     targets = self.dataloaders["train"].dataset.targets
    #     biases = []
    #     for b in self.biases:
    #         biases.append(self.dataloaders["train"].dataset[b])

    #     # Step 1: Stack all biases and create unique combinations
    #     biases = np.stack(biases, axis=1)  # Shape: (num_samples, num_biases)
    #     bias_labels = [
    #         tuple(bias) for bias in biases
    #     ]  # List of tuples for unique combinations
    #     unique_bias_labels = list(set(bias_labels))  # Unique bias combinations

    #     # Map each unique combination to a unique index
    #     bias_label_to_index = {
    #         label: idx for idx, label in enumerate(unique_bias_labels)
    #     }
    #     bias_indices = np.array(
    #         [bias_label_to_index[bias] for bias in bias_labels]
    #     )  # Bias indices

    #     # Step 2: Compute the confusion matrix
    #     conf_matrix = confusion_matrix(targets, bias_indices)

    #     self.criterion = BBLoss(conf_matrix)

    def _setup_criterion(self):
        super()._setup_criterion()
        # Example inputs
        all_targets = []
        all_biases = []

        if len(self.biases) > 1:
            raise ValueError("BB can be applied only for single attribute biases.")

        for data in self.dataloaders["train"]:

            # Collect targets
            all_targets.append(data["targets"])
            all_biases.append(data[self.biases[0]])

        # Concatenate all collected data
        all_targets = torch.cat(all_targets, dim=0)
        all_biases = torch.cat(all_biases, dim=0)

        # Step 2: Compute the confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_biases)
        # print(conf_matrix)
        self.criterion_train = BBLoss(torch.tensor(conf_matrix))

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}
