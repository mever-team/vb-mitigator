import os
from tools.utils import load_checkpoint, log_msg
import torch.nn.functional as F
import numpy as np
from tools.metrics.utils import AverageMeter
import torch
from tools.metrics import metrics_dicts, get_performance


from tqdm import tqdm
from .base_trainer import BaseTrainer
from models.builder import get_model
import torch.nn as nn
import torch
import copy
from tqdm import tqdm
from torch.utils.data import Subset

class PruneTrainer(BaseTrainer):

    def _setup_models(self):
        self.filter_activity = None
        self.target_layer = "extractor.7.1.conv2"  # change as needed
        self.num_filters_to_suppress = 400  # top-k filters
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        self.proj = nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        self.proj.to(self.device)
        # Register hook for collecting activations
        self.activations = []

        def hook_fn(module, input, output):
            # shape: [B, C, H, W] → mean over batch, H, W
            avg_act = output.detach().abs().mean(dim=(0, 2, 3))  # [C]
            self.activations.append(avg_act.cpu())

        layer = dict([*self.model.named_modules()])[self.target_layer]
        layer.register_forward_hook(hook_fn)

        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.model.to(self.device)

    def _collect_filter_statistics(self, dataloader, num_batches=100):
        self.model.eval()
        self.activations.clear()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                x = batch["inputs"].to(self.device)
                self.model(x)  # triggers hook

        all_acts = torch.stack(self.activations)  # [N_batches, C]
        mean_activity = all_acts.mean(dim=0)  # [C]
        self.filter_activity = mean_activity

    def _suppress_top_filters(self):
        if self.filter_activity is None:
            raise ValueError("Filter activity not collected.")

        topk_indices = torch.topk(self.filter_activity, self.num_filters_to_suppress).indices

        conv_layer = dict([*self.model.named_modules()])[self.target_layer]
        with torch.no_grad():
            conv_layer.weight[topk_indices] = 0.0
            if conv_layer.bias is not None:
                conv_layer.bias[topk_indices] = 0.0


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
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss
    

    # def _validate_epoch(self, stage="val"):
    #     self._set_eval()
    #     with torch.no_grad():
    #         all_data = {key: [] for key in self.biases}
    #         all_data["targets"] = []
    #         all_data["predictions"] = []

    #         losses = []
    #         for batch in self.dataloaders[stage]:
    #             batch_dict, loss = self._val_iter(batch)
    #             losses.append(loss.detach().cpu().numpy())
    #             for key, value in batch_dict.items():
    #                 all_data[key].append(value.detach().cpu().numpy())

    #         for key in all_data:
    #             all_data[key] = np.concatenate(all_data[key])
    #         # metric specific data
    #         if self.ba_groups is not None:
    #             all_data["ba_groups"] = self.ba_groups
    #         performance = get_performance[self.cfg.METRIC](all_data)
    #         performance["loss"] = np.mean(losses)
    #     return performance
    

    
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        with torch.no_grad():
            logits, features = self.model(inputs)
        
        features = self.proj(features.detach())

        # with torch.no_grad():
        logits2 = self.model.fc(features)

        # MAIN OPTIMIZER STEP
        self.optimizer.zero_grad()
        preds = torch.argmax(logits, dim=1)

        mask = preds != targets
        masked_logits = logits2[mask]
        masked_targets = targets[mask]

        loss_cls = self.criterion(masked_logits, masked_targets)

        logits_norm2 = ((logits2[range(logits2.shape[0]), targets]) ** 2).mean()
        logits_norm1 = ((logits[range(logits.shape[0]), targets]) ** 2).mean()
        # print(logits_norm1, logits_norm2)
        norm_loss = torch.abs(logits_norm1  - logits_norm2)
        # loss_norm = ((masked_logits[range(masked_logits.shape[0]), masked_targets]) ** 2).mean()

        # # STD REDUCTION LOSS: minimize std of logits per target class
        # unique_classes = torch.unique(masked_targets)
        # stds = []

        # for cls in unique_classes:
        #     cls_mask = masked_targets == cls
        #     cls_logits = masked_logits[cls_mask, cls]  # logits for true class
        #     if cls_logits.shape[0] > 1:
        #         std = torch.std(cls_logits)
        #         stds.append(std)

        # if stds:
        #     loss_norm = torch.stack(stds).mean()
        # else:
        #     loss_norm = torch.tensor(0.0, device=self.device)


        loss = loss_cls + norm_loss #torch.abs(loss_norm - 0.5)
        loss.backward()
        self.optimizer.step()

        return {"train_loss_main": loss_cls}
    

    def _setup_optimizer(self):
        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                self.proj.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=0.0,
            )
           
        elif self.cfg.SOLVER.TYPE == "Adam":
            self.optimizer = torch.optim.Adam(
                self.proj.parameters(),
                lr=self.cfg.SOLVER.LR,
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")


    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        logits1, features = self.model(inputs)
        features = self.proj(features)
        outputs = self.model.fc(features)
        outputs = 0.3 * outputs +  logits1
        loss = self.criterion(outputs, targets)
        # loss = ((outputs[range(outputs.shape[0]), targets]) ** 2).mean()

        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss
    
    def _set_train(self):
        self.model.eval()
        self.proj.train()

    def _set_eval(self):
        self.model.eval()
        self.proj.eval()

    def train(self):
        self.load_checkpoint("best_erm")
        # self._collect_filter_statistics(self.dataloaders["train"])
        # self._suppress_top_filters()
        start_epoch = self.current_epoch + 1
        for epoch in range(
            start_epoch,
            min(start_epoch + self.cfg.EXPERIMENT.EPOCH_STEPS, self.cfg.SOLVER.EPOCHS),
        ):
            self.current_epoch = epoch
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
        self._save_checkpoint(tag="latest")


    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
    
    def eval(self):
        self.load_checkpoint("best_erm")
        self.estimate_sample_influence()
        # test_performance = self._validate_epoch(stage="test")
        self._collect_filter_statistics(self.dataloaders["train"])
        self._suppress_top_filters()
        test_performance = self._validate_epoch(stage="test")
        test_log_dict = self.build_log_dict(test_performance, stage="test")
        print(log_msg(f"Test performance: {test_log_dict}", "EVAL", self.logger))




    def estimate_sample_influence(self):
        self.model.eval()

        # Save the initial state dict for resetting
        initial_state_dict = copy.deepcopy(self.model.state_dict())

        # Dataloader for full loss eval
        # full_loader = self.dataloaders["train"]

        # Original dataset
        full_dataset = self.sets["train"]

        # Select 10% of it
        total_len = len(full_dataset)
        subset_size = int(0.1 * total_len)
        subset_indices = np.random.choice(total_len, subset_size, replace=False)

        # Create a subset
        subset = Subset(full_dataset, subset_indices)

        # DataLoader
        full_loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False)


        # Compute initial total loss
        def compute_total_loss(model):
            model.eval()
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for batch in full_loader:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    logits, _ = model(inputs)

                    # mask = targets == 0
                    # targets = targets[mask]
                    # logits = logits[mask]
                    loss = self.criterion(logits, targets)
                    total_loss += loss.item() * targets.size(0)
                    count += targets.size(0)
            return total_loss / count
        
        def compute_per_sample_losses(model):
            losses = []
            model.eval()
            with torch.no_grad():
                for batch in full_loader:
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    logits, _ = model(inputs)

                    mask = targets == 0
                    targets = targets[mask]
                    logits = logits[mask]

                    loss = F.cross_entropy(logits, targets, reduction='none')  # returns tensor [batch_size]

                    losses.extend(loss.tolist())
            return losses
        
        initial_loss = compute_per_sample_losses(self.model)
        # print(f"initial loss: {initial_loss}")
        # Individual sample influence dict
        loss_diffs = {}

        one_loader = torch.utils.data.DataLoader(self.sets["train"], batch_size=1, shuffle=True)

        for batch in tqdm(one_loader):
            # Reload initial model state
            self.model.load_state_dict(copy.deepcopy(initial_state_dict))
            self.model.to(self.device)
            self.model.train()

            # Extract sample
            input_tensor = batch["inputs"].to(self.device)  # [1, ...]
            target_tensor = batch["targets"].to(self.device)
            if target_tensor[0] == 1:
                continue
            bias = batch[self.biases[0]]
            idx = batch["index"]

            # Optimizer
            optimizer =  torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )

            # for i in range(10):

            # One-step update
            optimizer.zero_grad()
            logits, _ = self.model(input_tensor)
            loss = self.criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()

            # Measure updated total loss
            updated_loss = compute_per_sample_losses(self.model)

            count_increased = sum(1 for init, updated in zip(initial_loss, updated_loss) if updated > init)
    # loss_diff = sum(initial_losses) - sum(updated_losses)
            # Store difference
            loss_diffs[idx[0].item()] = count_increased
            print(loss_diffs[idx[0].item()], bias[0].cpu().item(), target_tensor[0].cpu().item())
        torch.save(loss_diffs, "sample_loss_differences.pt")

        return 
