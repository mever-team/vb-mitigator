from tools.metrics.utils import AverageMeter
import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.3):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = ((1.0 - Yg.squeeze().detach())**self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss.mean()

class MultiTrainTrainer(BaseTrainer):

    # def _setup_criterion(self):
    #     self.criterion = GeneralizedCELoss(q=0.7)

    def create_crossval_splits(self):
        full_dataset = self.dataloaders["train"].dataset
        total_size = len(full_dataset)
        fold_size = total_size // 10

        indices = list(range(total_size))
        # torch.manual_seed(1234)
        indices = torch.randperm(total_size).tolist()  # shuffle

        self.crossval_splits = []
        for fold in range(10):
            start = fold * fold_size
            end = start + fold_size if fold < 9 else total_size  # last fold might have more samples

            mask_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]
            # print(len(train_indices), len(mask_indices))

            train_dataset = Subset(full_dataset, train_indices)
            mask_dataset = Subset(full_dataset, mask_indices)
            # mask_dataset = Subset(full_dataset, train_indices)
            # train_dataset = Subset(full_dataset, mask_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.dataloaders["train"].batch_size,
                shuffle=True
            )
            mask_loader = DataLoader(
                mask_dataset,
                batch_size=self.dataloaders["train"].batch_size,
                shuffle=False
            )

            self.crossval_splits.append({
                "train_loader": train_loader,
                "mask_loader": mask_loader
            })
            
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)
        indices = batch["index"].to(self.device)  # assuming batch contains sample indices
        mask_bc = biases !=targets 
        len_bc = torch.sum(mask_bc)

        mask_ba = ~mask_bc
        #select random len_bc samples from targets==biases
        # Indices of bias-aligned samples
        ba_indices = torch.where(mask_ba)[0]

        if len(ba_indices) >= len_bc:
            # Randomly sample len_bc bias-aligned examples
            selected_ba_indices = ba_indices[torch.randperm(len(ba_indices))[:len_bc]]
        else:
            # If not enough bias-aligned samples, take all (corner case)
            selected_ba_indices = ba_indices

        # Combine indices: all bias-conflicting + sampled bias-aligned
        selected_indices = torch.cat([torch.where(mask_bc)[0], selected_ba_indices])

        # Select inputs and targets
        inputs = inputs[selected_indices]
        targets = targets[selected_indices]

        # # Filter samples based on self.mask
        # if hasattr(self, "mask"):

        #     selected_mask = torch.tensor(
        #         [not self.mask.get(idx.item(), False) for idx in indices],
        #         device=self.device,
        #         dtype=torch.bool  # ensure it's a boolean mask
        #     )
        #     if selected_mask.sum() == 0:
        #         # Avoid empty batch crash: skip training step
        #         return {"train_cls_loss": torch.tensor(0.0, device=self.device)}
            
        #     inputs = inputs[selected_mask]
        #     targets = targets[selected_mask]
        #     indices = indices[selected_mask]
        

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        for batch in self.dataloaders["train"]:
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss

    def train(self):
        start_epoch = self.current_epoch + 1
        split = 0
        # self.mask = {}
        # self.create_crossval_splits()
        for epoch in range(
            start_epoch,
            min(start_epoch + self.cfg.EXPERIMENT.EPOCH_STEPS, self.cfg.SOLVER.EPOCHS),
        ):
            self.current_epoch = epoch
            # if epoch == 1:
            #     self.dataloaders["train_main"] = self.crossval_splits[0]["train_loader"]
            #     self.dataloaders["train_mask"] = self.crossval_splits[0]["mask_loader"]
            log_dict = self._train_epoch()
            # log_dict = {}
            if self.cfg.LOG.TRAIN_PERFORMANCE:
                if self.cfg.METRIC == "wg_ovr_tags":
                    raise ValueError(
                        f"the self.cfg.LOG.TRAIN_PERFORMANCE should be False for wg_ovr_tags metric, you gave {self.cfg.LOG.TRAIN_PERFORMANCE}"
                    )
                train_performance = self._validate_epoch(stage="train")
                train_log_dict = self.build_log_dict(train_performance, stage="train")
                log_dict.update(train_log_dict)
            if self.cfg.LOG.SAVE_CRITERION == "val":
                if self.cfg.METRIC == "wg_ovr_tags":
                    raise ValueError(
                        f"the self.cfg.LOG.SAVE_CRITERION should be test for wg_ovr_tags metric, you gave {self.cfg.LOG.SAVE_CRITERION}"
                    )
                val_performance = self._validate_epoch(stage="val")
                val_log_dict = self.build_log_dict(val_performance, stage="val")
                log_dict.update(val_log_dict)
            if self.cfg.METRIC == "wg_ovr_tags":
                test_performance = self._validate_epoch_tags(stage="test")
            else:
                test_performance = self._validate_epoch(stage="test")
            test_log_dict = self.build_log_dict(test_performance, stage="test")
            log_dict.update(test_log_dict)
            update_cpkt = self._update_best(log_dict)
            if update_cpkt:
                self._save_checkpoint(tag="best")
            self._save_checkpoint(tag=f"current_{self.cfg.EXPERIMENT.SEED}")
            self._log_epoch(log_dict, update_cpkt)


            # if epoch %2 == 0 and split <9:
            #     self.mask = self.compute_mask(mask=self.mask)

            #     split+=1 
            #     self.dataloaders["train_main"] = self.crossval_splits[split]["train_loader"]
            #     self.dataloaders["train_mask"] = self.crossval_splits[split]["mask_loader"]
                
            #     self._setup_models()
            #     self._setup_optimizer()
            #     self._setup_scheduler()
            # elif split == 9:
            #     self.dataloaders["train_main"] = self.dataloaders["train"] 
            #     self.mask = {k: not v for k, v in self.mask.items()}

            #     self._setup_models()
            #     self._setup_optimizer()
            #     self._setup_scheduler()

        self._save_checkpoint(tag="latest")


    def compute_mask(self, mask, confidence_threshold=0.9):
        self.model.eval()
        # mask = {}  # index -> boolean (True if high confidence misclassified)
        all_selected = []
        all_correct = []

        with torch.no_grad():
            for batch in self.dataloaders["train_mask"]:  # <-- changed here
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                biases = batch[self.biases[0]].to(self.device)
                indices = batch["index"]
                
                logits, _ = self.model(inputs)
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                preds_confidence = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

                misclassified = preds != targets
                high_confidence = preds_confidence >= confidence_threshold
                selected = misclassified & high_confidence
                correct = (targets != biases)

                all_selected.append(selected)
                all_correct.append(correct)

                for idx, flag in zip(indices, selected):
                    mask[idx.item()] = flag.item()

        all_selected = torch.cat(all_selected)
        all_correct = torch.cat(all_correct)

        if all_selected.sum() > 0:
            precision = (all_correct[all_selected].float().mean()).item()
            print(f"Mask Precision: {precision:.4f} , num samples: {torch.sum(all_selected).item()}, total bc samples: {torch.sum(all_correct).item()}")
        else:
            print("No samples selected (precision undefined).")

        return mask