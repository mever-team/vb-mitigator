# Based on the implementation provided by https://github.com/facebookresearch/Whac-A-Mole/blob/main/urbancars_trainers/jtt.py

import os
from .base_trainer import BaseTrainer
import torch
from torch.utils.data import Subset, ConcatDataset
from models.utils import get_local_model_dict


class JTTTrainer(BaseTrainer):

    def bias_detection(self):
        cfg = self.cfg
        self.model.train()

        erm_id_optimizer = torch.optim.SGD(
            self.model.parameters(),
            cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        MODEL_SAVE_PATH = os.path.join(self.log_path, "bias_discovery_model")
        if cfg.MITIGATOR.JTT.BIAS_DISCOVERY_EPOCHS == 0:
            self.model = get_local_model_dict(self.cfg.MITIGATOR.JTT.BCC_PATH)
            print(
                f"loaded vinilla model as bias discovery model ({cfg.MITIGATOR.JTT.BCC_PATH})"
            )
        elif os.path.exists(MODEL_SAVE_PATH):
            print("Loading pre-trained bias detection model...")
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
            self.model.to(self.device)
        else:
            print(
                f"training for {cfg.MITIGATOR.JTT.BIAS_DISCOVERY_EPOCHS} epoch(s) to detect the biases."
            )
            for epoch in range(cfg.MITIGATOR.JTT.BIAS_DISCOVERY_EPOCHS):
                total_loss = 0.0
                correct = 0
                total = 0

                for batch in self.dataloaders["train"]:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)

                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs
                    
                    loss = self.criterion(outputs, targets)

                    # Backpropagation
                    self._loss_backward(loss)
                    erm_id_optimizer.step()
                    erm_id_optimizer.zero_grad(set_to_none=True)

                    # Track loss
                    total_loss += loss.item()

                    # Track accuracy
                    predicted = outputs.argmax(dim=1)  # Assuming classification task
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

                avg_loss = total_loss / len(self.dataloaders["train"])
                accuracy = correct / total * 100 if total > 0 else 0

                print(f"Epoch [{epoch+1}/{cfg.MITIGATOR.JTT.BIAS_DISCOVERY_EPOCHS}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


            os.makedirs("saved_models", exist_ok=True)
            torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved at {MODEL_SAVE_PATH}")
        self.model.eval()

        error_set_list = []

        ordered_train_loader = torch.utils.data.DataLoader(
            self.sets["train"],
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,  # no shuffle for inferring error set
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
        )

        for batch in ordered_train_loader:
            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
            pred = outputs.argmax(dim=1)
            error = pred != targets
            error_set_list.append(error.long().cpu())

        error_set_list = torch.cat(error_set_list, dim=0)
        error_indices = torch.nonzero(error_set_list).flatten().tolist()

        train_set = self.sets["train"]
        upsampled_points = Subset(train_set, error_indices * cfg.MITIGATOR.JTT.UPWEIGHT)
        concat_train_set = ConcatDataset([train_set, upsampled_points])
        train_loader = torch.utils.data.DataLoader(
            concat_train_set,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,  # no shuffle for inferring error set
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
        )
        self.dataloaders["train"] = train_loader

        self._setup_models()
        self._setup_optimizer()

    def _method_specific_setups(self):
        self.bias_detection()
