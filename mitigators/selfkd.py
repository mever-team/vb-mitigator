import os
import torch
from configs.cfg import show_cfg
from models.builder import get_model
from tools.utils import seed_everything
from .base_trainer import BaseTrainer
from .losses import DistillKL


class SelfkdTrainer(BaseTrainer):

    def __init__(self, cfg):
        seed_everything(cfg.EXPERIMENT.SEED)
        self.cfg = cfg
        self._setup_logger()
        self.device = self._setup_device()
        self.current_epoch = 0
        self._method_specific_setups()
        self._setup_dataset()
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._metric_specific_setups()
        self._basic_eval_setups()

        # self._setup_resume()
        show_cfg(cfg, self.logger)

    def _method_specific_setups(self):
        self.teacher_path = self.cfg.MITIGATOR.SELFKD.TEACHER_PATH
        self.rep = 1
        return

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, self.cfg.MODEL.PRETRAINED
        )
        self.model.to(self.device)

        self.teacher = get_model(
            self.cfg.MODEL.TYPE, self.num_class, self.cfg.MODEL.PRETRAINED
        )
        self.teacher.load_state_dict(torch.load(self.teacher_path)["model"])
        self.teacher.to(self.device)
        self.teacher.eval()

    def _setup_criterion(self):
        self.criterion_kd = DistillKL(T=4)
        return super()._setup_criterion()

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        with torch.no_grad():
            t_logits, t_feats = self.teacher(inputs)

        outputs, _ = self.model(inputs)
        preds = torch.argmax(t_logits, dim=1)

        mask = preds == targets
        loss_kd = self.criterion_kd(outputs[mask], t_logits[mask])
        if torch.sum(~mask) == 0:
            loss_cl = torch.tensor(0.0).to(self.device)
            return {
                "train_kd_loss": torch.tensor(0.0).to(self.device),
                "train_cls_loss": loss_cl,
            }
        else:
            loss_cl = self.criterion(outputs[~mask], targets[~mask])
        loss = loss_cl
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_kd_loss": loss_kd, "train_cls_loss": loss_cl}

    def train(self):
        if self.rep > 5:
            return
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

        self.rep += 1
        self.current_epoch = 0
        self.teacher_path = os.path.join(self.log_path, "best")
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self.train()
