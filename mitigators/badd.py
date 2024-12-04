"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# --------------------------------------------------------
# implementation from LfF:
# https://github.com/alinlab/LfF
# --------------------------------------------------------

import torch
import torch.nn as nn


from utils import IdxDataset, EMAGPU as EMA
from tqdm import tqdm
from .base_trainer import BaseTrainer
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler


class BAddResNet(models.ResNet):
    def __init__(self):
        super(BAddResNet, self).__init__(Bottleneck, [3, 4, 6, 3])

    def concat_forward(self, x, f):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = F.normalize(x, dim=1)
        # f = F.normalize(f, dim=1)
        feats = x + f
        x = self.fc(feats)
        # norm_x = torch.norm(feats, dim=1)
        # norm_f = torch.norm(f, dim=1)
        # print("Norm of x:", norm_x)
        # print("Norm of f:", norm_f)

        return x, feats

    def concat_forward2(self, x, f, f2):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # norm_x = torch.norm(x, dim=1)
        # norm_f = torch.norm(f, dim=1)
        # print("Norm of x:", norm_x)
        # print("Norm of f:", norm_f)

        # x = F.normalize(x, dim=1)
        # f = F.normalize(f, dim=1)
        # f2 = F.normalize(f2, dim=1)
        feats = x + f + f2  # * 100
        x = self.fc(feats)

        return x, feats

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(x.type())
        # print(self.conv1.weight.type())
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = F.normalize(x, dim=1)
        x = self.fc(x)

        return x


class BAddResNet18(models.ResNet):
    def __init__(self):
        super(BAddResNet18, self).__init__(Bottleneck, [3, 4, 6, 3])

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class BAddTrainer(BaseTrainer):
    def _method_specific_setups(self):
        train_target_attr = self.train_set.get_labels()[:, 0]

    # def _modify_train_set(self, train_dataset):
    #     return IdxDataset(train_dataset)

    # def _get_train_loader(self, train_set):
    #     args = self.args
    #     weights = train_set.get_sampling_weights()
    #     sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
    #     train_loader = torch.utils.data.DataLoader(
    #         train_set,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_memory,
    #         sampler=sampler,
    #         persistent_workers=args.num_workers > 0,
    #     )
    #     return train_loader

    def _setup_models(self):
        super(BAddTrainer, self)._setup_models()
        # print(self.classifier.conv1.weight)
        self.classifier2 = BAddResNet()
        self.classifier2.fc = nn.Linear(self.classifier.fc.in_features, 2)
        self.classifier2.load_state_dict(self.classifier.state_dict())
        self.classifier = self.classifier2
        self.classifier = self.classifier.to(self.device)
        # print(self.classifier.conv1.weight)
        b_labels_len = self.train_set.get_labels()
        b_labels_len = b_labels_len.shape[1]
        self.bias_discover_net = nn.Linear(2, self.classifier.fc.in_features).to(
            self.device
        )
        self.bias_discover_net2 = nn.Linear(2, self.classifier.fc.in_features).to(
            self.device
        )

        # self.bias_discover_net = BAddResNet18()
        # self.bias_discover_net.fc = nn.Linear(self.bias_discover_net.fc.in_features, 2)
        # self.bias_discover_net.load_state_dict(torch.load("bcc_places.pth"))
        # self.bias_discover_net = self.bias_discover_net.to(self.device)

        # self.bias_discover_net2 = BAddResNet18()
        # self.bias_discover_net2.fc = nn.Linear(
        #     self.bias_discover_net2.fc.in_features, 2
        # )
        # self.bias_discover_net2.load_state_dict(torch.load("bcc_obj.pth"))
        # self.bias_discover_net2 = self.bias_discover_net2.to(self.device)

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "badd"
        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def train(self):
        args = self.args
        self.bias_discover_net.eval()
        self.bias_discover_net2.eval()
        self.classifier.train()

        total_cls_loss = 0
        total_ce_loss = 0

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for idx, data_dict in enumerate(pbar):
            img, all_attr_label = data_dict["image"], data_dict["label"]
            img = img.to(self.device, non_blocking=True)
            label = all_attr_label[:, 0]
            label = label.to(self.device, non_blocking=True)
            # print(all_attr_label.shape)
            b1 = all_attr_label[:, 1]
            b2 = all_attr_label[:, 2]
            b1 = b1.to(self.device, non_blocking=True)
            b2 = b2.to(self.device, non_blocking=True)

            b1 = F.one_hot(b1, num_classes=2).float()
            b2 = F.one_hot(b2, num_classes=2).float()
            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    # spurious_features = self.bias_discover_net.get_features(img)
                    # spurious_features2 = self.bias_discover_net2.get_features(img)
                    spurious_features = self.bias_discover_net(b1)
                    spurious_features2 = self.bias_discover_net2(b2)

                # print(img.type())
                # target_logits = self.classifier(img)
                target_logits, _ = self.classifier.concat_forward2(
                    img, spurious_features, spurious_features2
                )
                # tl = self.classifier(img)
                ce_loss = self.criterion(target_logits, label)
                # ce_loss2 = self.criterion(tl, label)

            loss = ce_loss  # + 0.01 * ce_loss2

            self.optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)

            self._scaler_update()

            total_cls_loss += loss.item()
            total_ce_loss += ce_loss.item()
            avg_cls_loss = total_cls_loss / (idx + 1)
            avg_ce_loss = total_ce_loss / (idx + 1)

            pbar.set_description(
                "[{}/{}] cls_loss: {:.3f}, ce: {:.3f}".format(
                    self.cur_epoch,
                    args.epoch,
                    avg_cls_loss,
                    avg_ce_loss,
                )
            )

        log_dict = {
            "loss": total_cls_loss / len(self.train_loader),
            "ce_loss": total_ce_loss / len(self.train_loader),
        }
        self.log_to_wandb(log_dict)

    def _state_dict_for_save(self):
        state_dict = super(BAddTrainer, self)._state_dict_for_save()
        return state_dict

    def _load_state_dict(self, state_dict):
        super(BAddTrainer, self)._load_state_dict(state_dict)
