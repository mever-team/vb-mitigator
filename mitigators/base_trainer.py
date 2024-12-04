import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from datasets.builder import get_dataset
from models.builder import get_model
from tools.metrics import metrics_dicts, get_performance

from tools.utils import log_msg, save_checkpoint, load_checkpoint, setup_logger
from configs.cfg import show_cfg


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._setup_logger()
        self.device = self._setup_device()
        self.current_epoch = 0
        self._setup_dataset()
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._metric_specific_setups()
        self._method_specific_setups()
        show_cfg(cfg, self.logger)

    def _setup_device(self):
        return torch.device(
            self.cfg.EXPERIMENT.GPU if torch.cuda.is_available() else "cpu"
        )

    def _metric_specific_setups(self):
        self.metric_dict = metrics_dicts[self.cfg.METRIC]
        if self.metric_dict["best"] == "high":
            self.best_performance = float("-inf")
        elif self.metric_dict["best"] == "low":
            self.best_performance = float("inf")
        else:
            raise ValueError(
                f"Unsupported metric best type: {self.metric_dict['best']}"
            )

    def _setup_dataset(self):
        dataset = get_dataset(self.cfg)
        self.num_class = dataset["num_class"]
        self.biases = dataset["biases"]
        self.dataloaders = dataset["dataloaders"]
        # self.batch_structure = dataset["batch_structure"]

    def _setup_optimizer(self):
        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _loss_backward(self, loss):
        loss.backward()

    def _optimizer_step(self):
        self.optimizer.step()

    def _set_train(self):
        self.model.train()

    def _set_eval(self):
        self.model.eval()

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
        elif self.cfg.SOLVER.SCHEDULER.TYPE == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.SOLVER.SCHEDULER.T_MAX
            )
        else:
            raise ValueError(
                f"Unsupported scheduler type: {self.cfg.SOLVER.SCHEDULER.TYPE}"
            )

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        for batch in self.dataloaders["train"]:
            self._train_iter(batch)
        self.scheduler.step()

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

    def _validate_epoch(self, stage="val"):
        self._set_eval()
        with torch.no_grad():
            all_data = {key: [] for key in self.biases}
            all_data["targets"] = []
            all_data["predictions"] = []
            losses = []
            for batch in self.dataloaders[stage]:
                batch_dict, loss = self._val_iter(batch)
                losses.append(loss.detach().cpu().numpy())
                for key, value in batch_dict.items():
                    all_data[key].append(value.detach().cpu().numpy())

            for key in all_data:
                all_data[key] = np.concatenate(all_data[key])
            performance = get_performance[self.cfg.METRIC](all_data)
            performance["loss"] = np.mean(losses)
        return performance

    def _method_specific_setups(self):
        pass

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.model.to(self.device)

    def _setup_logger(self):
        tags = self.cfg.EXPERIMENT.TAG.split(",")
        experiment_name = os.path.join(
            self.cfg.EXPERIMENT.PROJECT,
            self.cfg.EXPERIMENT.TAG,
            self.cfg.EXPERIMENT.NAME,
        )
        self.log_path = os.path.join(self.cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger = setup_logger(os.path.join(self.log_path, "out.log"))
        if self.cfg.LOG.WANDB:
            try:
                import wandb

                wandb.init(
                    project=self.cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags
                )
            except ImportError as e:
                print(log_msg(f"Failed to import WANDB: {e}", "INFO", self.logger))
                self.cfg.LOG.WANDB = False
            except Exception as e:
                print(log_msg(f"Failed to use WANDB: {e}", "INFO", self.logger))
                self.cfg.LOG.WANDB = False

        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def build_log_dict(self, p_dict, stage="train"):
        log_dict = {}
        for key, value in p_dict.items():
            log_dict[f"{stage}_{key}"] = value
        return log_dict

    def _log_epoch(self, log_dict, update_cpkt):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, self.current_epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": self.current_lr})
            wandb.log(log_dict)
        if update_cpkt:
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_performance"] = self.best_performance
        # worklog.txt
        with open(os.path.join(self.log_path, "out.log"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(self.current_epoch) + os.linesep,
                "lr: {:.6f}".format(float(self.current_lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def _update_best(self, log_dict):
        if (
            log_dict[f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"]
            > self.best_performance
            and self.metric_dict["best"] == "high"
        ):
            self.best_performance = log_dict[
                f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"
            ]
            return True
        if (
            log_dict[f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"]
            < self.best_performance
            and self.metric_dict["best"] == "low"
        ):
            self.best_performance = log_dict[
                f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"
            ]
            return True
        return False

    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_performance": self.best_performance,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )

    def train(self):
        for epoch in range(self.cfg.SOLVER.EPOCHS):
            self.current_epoch = epoch
            self._train_epoch()
            log_dict = {}
            if self.cfg.LOG.TRAIN_PERFORMANCE:
                train_performance = self._validate_epoch(stage="train")
                train_log_dict = self.build_log_dict(train_performance, stage="train")
                log_dict.update(train_log_dict)
            if self.cfg.LOG.SAVE_CRITERION == "val":
                val_performance = self._validate_epoch(stage="val")
                val_log_dict = self.build_log_dict(val_performance, stage="val")
                log_dict.update(val_log_dict)
            test_performance = self._validate_epoch(stage="test")
            test_log_dict = self.build_log_dict(test_performance, stage="test")
            log_dict.update(test_log_dict)
            update_cpkt = self._update_best(log_dict)
            if update_cpkt:
                self._save_checkpoint(tag="best")
            self._log_epoch(log_dict, update_cpkt)
        self._save_checkpoint(tag="latest")

    def eval(self):
        self.load_checkpoint(self.cfg.LOG.SAVE_CRITERION)
        test_performance = self._validate_epoch(stage="test")
        test_log_dict = self.build_log_dict(test_performance, stage="test")
        print(log_msg(f"Test performance: {test_log_dict}", "EVAL", self.logger))
