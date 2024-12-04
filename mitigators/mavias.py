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

import csv
import torch
import torch.nn as nn


from utils import IdxDataset, EMAGPU as EMA
from tqdm import tqdm
from .base_trainer import BaseTrainer
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from .losses import SupConLoss
from dataset.urbancars import UrbanCars
from dataset.urbancars_clip import UrbanCarsClip
from model.classifiers import (
    get_classifier,
    get_transforms,
)
from utils import (
    set_seed,
    MultiDimAverageMeter,
)
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
from model.mixup_cutmix_transforms import RandomMixup, RandomCutmix
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Dense layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        # print("you are here")
        # x = self.relu(x)
        # x = self.fc2(x)
        return x


# def precompute_text_embeddings(device, tags_voc, tokenizer, model):
#     precomputed_embeddings = []
#     with torch.no_grad():
#         for idx in range(len(tags_voc)):
#             # print(comma_separated_tags)
#             text_inputs = tokenizer(
#             [tags_voc[idx]],
#             padding="max_length",
#             return_tensors="pt",
#             ).to(device)
#             b_feats = model.get_text_features(**text_inputs)
#             for prompt, emb in zip([tags_voc[idx]], b_feats):
#                 precomputed_embeddings.append(emb.detach().cpu())
#     return torch.stack(precomputed_embeddings)
def precompute_text_embeddings(trainloader, device, tags_voc, tokenizer, model):
    precomputed_embeddings = {}
    with torch.no_grad():
        for batch_idx, (_, _, bias, _) in enumerate(tqdm(trainloader)):

            bias = bias.to(device)
            # print(bias)
            # Find the indices of non-zero elements (tag presence)
            sample_indices, tag_indices = torch.nonzero(bias, as_tuple=True)

            # Map the non-zero indices to tag names
            tags_for_samples = [tags_voc[idx] for idx in tag_indices]

            # Group tags by sample
            samples_tags = {i.item(): [] for i in torch.unique(sample_indices)}
            for sample, tag in zip(sample_indices, tags_for_samples):
                samples_tags[sample.item()].append(tag)
                # Ensure every sample index has an entry in the dictionary
            for i in range(bias.shape[0]):
                if i not in samples_tags:
                    samples_tags[i] = []
            # print(samples_tags)
            comma_separated_tags = [
                "a photo with " + ", ".join(samples_tags[i])
                for i in range(bias.shape[0])
            ]
            # print(comma_separated_tags)
            text_inputs = tokenizer(
                comma_separated_tags,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            b_feats = model.get_text_features(**text_inputs)
            for prompt, emb in zip(comma_separated_tags, b_feats):
                precomputed_embeddings[prompt] = emb.detach().cpu()
    return precomputed_embeddings


class BAddResNet(models.ResNet):
    def __init__(self):
        super(BAddResNet, self).__init__(Bottleneck, [3, 4, 6, 3])
        # super(BAddResNet, self).__init__(BasicBlock, [2,2,2,2])

    def concat_forward(self, x, f, m):
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

        # feats = x + m*f
        # x = self.fc(feats)

        x = self.fc(x)
        fx = self.fc(f)
        out = x + fx
        return out, [x, fx]

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


class MAViasTrainer(BaseTrainer):

    def _setup_dataset(self):
        args = self.args

        train_transform = self._get_train_transform()
        test_transform = get_transforms(args.arch, is_training=False)

        train_set = UrbanCarsClip(
            transform=train_transform,
        )
        self.train_set = train_set
        val_set = UrbanCars(
            "data",
            "val",
            transform=test_transform,
        )
        test_set = UrbanCars(
            "data",
            "test",
            transform=test_transform,
        )
        # self.obj_name_list = train_set.obj_name_list
        self.num_class = 2

        train_set = self._modify_train_set(train_set)
        train_loader = self._get_train_loader(train_set)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tags_voc = train_set.tags_utility.unique_tags

    def _setup_optimizers(self):
        super(MAViasTrainer, self)._setup_optimizers()
        parameters_projection = self.proj_net.parameters()
        self.optimizer_projection = torch.optim.SGD(
            parameters_projection, lr=0.001, momentum=0.9, weight_decay=5e-4
        )

    def _setup_models(self):
        super(MAViasTrainer, self)._setup_models()
        self.classifier2 = BAddResNet()
        self.classifier2.fc = nn.Linear(self.classifier.fc.in_features, 2)
        self.classifier2.load_state_dict(self.classifier.state_dict())
        self.classifier = self.classifier2
        self.classifier = self.classifier.to(self.device)
        self.proj_net = SimpleMLP(
            768, self.classifier.fc.in_features, self.classifier.fc.in_features
        )
        self.proj_net = self.proj_net.to(self.device)
        # for p in self.classifier.parameters():
        #     p.requires_grad = False

        # for p in self.classifier.fc.parameters():
        #     p.requires_grad = True
        # for p in self.classifier.layer4.parameters():
        #     p.requires_grad = True

        # proj_net.train()

        models = [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]

        model_id = models[2]

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(self.device)
        self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.precomputed_embeddings = precompute_text_embeddings(
            trainloader=self.train_loader,
            device=self.device,
            tags_voc=self.tags_voc,
            tokenizer=self.tokenizer,
            model=self.clip_model,
        )
        # self.precomputed_embeddings = precompute_text_embeddings( device=self.device, tags_voc=self.tags_voc, tokenizer=self.tokenizer, model=self.clip_model)
        # self.precomputed_embeddings = self.precomputed_embeddings.to(self.device)

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "clip"
        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

        # CSV file name
        self.csv_file = "training_data.csv"

        # Write header to CSV
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "subgroup",
                    "main_logits",
                    "main_acc",
                    "clip_logits",
                    "clip_acc",
                    "combined_logits",
                    "combined_acc",
                ]
            )

    def train(self):
        args = self.args
        self.classifier.train()
        train_proj_net = self.cur_epoch < 600
        if train_proj_net:
            self.proj_net.train()
        else:
            self.proj_net.eval()

        total_ce_loss = 0
        total_norm_loss = 0

        total_norm_main = 0
        total_norm_clip = 0

        # total_main_logits = {'000':[0,0], '001':[0,0], '010':[0,0], '011':[0,0], '100':[0,0], '101':[0,0], '110':[0,0], '111':[0,0]}
        # total_clip_logits = {'000':[0,0], '001':[0,0], '010':[0,0], '011':[0,0], '100':[0,0], '101':[0,0], '110':[0,0], '111':[0,0]}
        # total_combined_logits = {'000':[0,0], '001':[0,0], '010':[0,0], '011':[0,0], '100':[0,0], '101':[0,0], '110':[0,0], '111':[0,0]}

        # total_main_correct = {'000':0, '001':0, '010':0, '011':0, '100':0, '101':0, '110':0, '111':0}
        # total_clip_correct = {'000':0, '001':0, '010':0, '011':0, '100':0, '101':0, '110':0, '111':0}
        # total_combined_correct = {'000':0, '001':0, '010':0, '011':0, '100':0, '101':0, '110':0, '111':0}
        # total_samples = {'000':0, '001':0, '010':0, '011':0, '100':0, '101':0, '110':0, '111':0}

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for idx, (images, targets, bias, bg_cooc) in enumerate(pbar):
            images, targets, bias = (
                images.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
                bias.to(self.device, non_blocking=True),
            )
            bg_labels = bg_cooc[0].to(self.device, non_blocking=True)
            cooc_labels = bg_cooc[1].to(self.device, non_blocking=True)

            # weighted_sum = torch.mm(bias, self.precomputed_embeddings)  # Shape: (batch_size, num_features)
            # # Calculate the number of tags present in each sample
            # num_present_tags = torch.sum(bias, dim=1, keepdim=True)  # Shape: (batch_size, 1)
            # # Avoid division by zero by replacing 0 with 1 in num_present_tags
            # num_present_tags = torch.where(num_present_tags == 0, torch.tensor(1.0, device=self.device), num_present_tags)
            # # Calculate the average features
            # b_feats = weighted_sum / num_present_tags  # Shape: (batch_size, num_features)

            # Find the indices of non-zero elements (tag presence)
            sample_indices, tag_indices = torch.nonzero(bias, as_tuple=True)

            # Map the non-zero indices to tag names
            tags_for_samples = [self.tags_voc[idx] for idx in tag_indices]

            # Group tags by sample
            samples_tags = {i.item(): [] for i in torch.unique(sample_indices)}
            for sample, tag in zip(sample_indices, tags_for_samples):
                samples_tags[sample.item()].append(tag)
                # Ensure every sample index has an entry in the dictionary
            for i in range(bias.shape[0]):
                if i not in samples_tags:
                    samples_tags[i] = []

            b_feats = torch.stack(
                [
                    self.precomputed_embeddings[
                        "a photo with " + ", ".join(samples_tags[i])
                    ]
                    for i in range(bias.shape[0])
                ]
            )

            b_feats = b_feats.to(self.device)
            if train_proj_net:
                b_feats = self.proj_net(b_feats)
            else:
                with torch.no_grad():
                    b_feats = self.proj_net(b_feats)

            # logits, features = self.classifier(images)
            # if self.cur_epoch<=8:
            #     m =10
            # else:
            #     m=0
            # m = 100/self.cur_epoch
            logits, feats = self.classifier.concat_forward(images, b_feats, 1)
            tmp = feats[1].detach().cpu().clone()
            norm_main = torch.norm(feats[0])
            norm_clip = torch.norm(tmp).to(self.device)
            norm_loss = F.mse_loss(norm_main, norm_clip * 0.4)
            # print(torch.norm(feats[0]), torch.norm(tmp))
            ce_loss = self.criterion(logits, targets)

            # mask = ((targets == 0) &  (bg_labels == 1) & (cooc_labels == 1)) | ((targets == 1) &  (bg_labels == 0) & (cooc_labels == 0))
            # m_logits = main_logits[range(logits.shape[0]), targets]
            # m_logits = main_logits[mask]

            # norm_loss = (m_logits ** 2).mean()

            loss = ce_loss + 0.01 * norm_loss

            self.optimizer.zero_grad(set_to_none=True)
            if train_proj_net:
                self.optimizer_projection.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            if train_proj_net:
                self._optimizer_step(self.optimizer_projection)

            self._scaler_update()

            total_ce_loss += ce_loss.item()
            total_norm_loss += norm_loss.item()
            total_norm_clip += norm_clip.item()
            total_norm_main += norm_main.item()
            avg_ce_loss = total_ce_loss / (idx + 1)
            avg_norm_loss = total_norm_loss / (idx + 1)
            avg_norm_clip = total_norm_clip / (idx + 1)
            avg_norm_main = total_norm_main / (idx + 1)

            # for key in total_main_logits.keys():
            #     key_target = int(key[0])
            #     key_bg = int(key[1])
            #     key_cooc = int(key[2])
            #     # print(targets.shape, bg_labels.shape, cooc_labels.shape)
            #     if targets.shape != bg_labels.shape or targets.shape != cooc_labels.shape:

            #         raise ValueError("The shapes of targets, bg_labels, and cooc_labels must match")

            #     mask_target = targets == key_target
            #     mask_bg = bg_labels == key_bg
            #     mask_cooc = cooc_labels == key_cooc
            #     mask = mask_target & mask_bg & mask_cooc

            #     main_logits = feats[0]
            #     clip_logits = feats[1]

            #     main_logits = main_logits[mask]
            #     clip_logits = clip_logits[mask]
            #     combined_logits = logits[mask]
            #     targets_m = targets[mask]

            #     total_main_logits[key][0] += torch.sum(main_logits,dim=0)[0].detach().cpu().item()
            #     total_main_logits[key][1] += torch.sum(main_logits,dim=0)[1].detach().cpu().item()

            #     total_clip_logits[key][0] += torch.sum(clip_logits,dim=0)[0].detach().cpu().item()
            #     total_clip_logits[key][1] += torch.sum(clip_logits,dim=0)[1].detach().cpu().item()

            #     total_combined_logits[key][0] += torch.sum(combined_logits,dim=0)[0].detach().cpu().item()
            #     total_combined_logits[key][1] += torch.sum(combined_logits,dim=0)[1].detach().cpu().item()

            #     pred_main = main_logits.argmax(dim=1)
            #     pred_clip = clip_logits.argmax(dim=1)
            #     pred_combined = combined_logits.argmax(dim=1)

            #     correct_main = torch.sum(pred_main == targets_m)
            #     correct_clip = torch.sum(pred_clip == targets_m)
            #     correct_combined = torch.sum(pred_combined == targets_m)

            #     total_main_correct[key] += correct_main
            #     total_clip_correct[key]  += correct_clip
            #     total_combined_correct[key]  += correct_combined
            #     total_samples[key]  += targets_m.shape[0]

            pbar.set_description(
                "[{}/{}] ce: {:.3f}, norm: {:.3f}, main_norm: {:.3f}, clip_norm: {:.3f}".format(
                    self.cur_epoch,
                    args.epoch,
                    avg_ce_loss,
                    avg_norm_loss,
                    avg_norm_main,
                    avg_norm_clip,
                )
            )

        # for key in total_main_logits.keys():
        #     main_logits = [x / total_samples[key] for x in total_main_logits[key]]
        #     clip_logits = [x / total_samples[key] for x in total_clip_logits[key]]
        #     combined_logits = [x / total_samples[key] for x in total_combined_logits[key]]
        #     main_acc = total_main_correct[key] / total_samples[key]
        #     clip_acc = total_clip_correct[key] / total_samples[key]
        #     combined_acc = total_combined_correct[key] / total_samples[key]

        #     # Append data to CSV
        #     with open(self.csv_file, mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([self.cur_epoch, key, main_logits, main_acc.detach().cpu().item(), clip_logits, clip_acc.detach().cpu().item(), combined_logits, combined_acc.detach().cpu().item()])

        #     print(f"subgroup: {key}, #samples: {total_samples[key]}")
        #     print(f"[{key}, main] logits: {[x / total_samples[key] for x in total_main_logits[key]]}, acc: {total_main_correct[key]/total_samples[key]}")
        #     print(f"[{key}, clip] logits: {[x / total_samples[key] for x in total_clip_logits[key]]}, acc: {total_clip_correct[key]/total_samples[key]}")
        #     print(f"[{key}, combined] logits: {[x / total_samples[key] for x in total_combined_logits[key]]}, acc: {total_combined_correct[key]/total_samples[key]}")
        log_dict = {
            "ce_loss": total_ce_loss / len(self.train_loader),
            "norm_loss": total_norm_loss / len(self.train_loader),
        }
        self.log_to_wandb(log_dict)

    def _state_dict_for_save(self):
        state_dict = super(MAViasTrainer, self)._state_dict_for_save()
        return state_dict

    def _load_state_dict(self, state_dict):
        super(MAViasTrainer, self)._load_state_dict(state_dict)

    @torch.no_grad()
    def _eval_split(self, loader, split):
        args = self.args

        meter = MultiDimAverageMeter((self.num_class, self.num_class, self.num_class))
        total_correct = []
        total_bg_correct = []
        total_co_occur_obj_correct = []
        total_shortcut_conflict_mask = []

        self.classifier.eval()
        pbar = tqdm(loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(image)

            pred = output.argmax(dim=1)

            obj_label = target[:, 0]
            bg_label = target[:, 1]
            co_occur_obj_label = target[:, 2]

            shortcut_conflict_mask = bg_label != co_occur_obj_label
            total_shortcut_conflict_mask.append(shortcut_conflict_mask.cpu())

            correct = pred == obj_label
            meter.add(correct.cpu(), target.cpu())
            total_correct.append(correct.cpu())

            bg_correct = pred == bg_label
            total_bg_correct.append(bg_correct.cpu())

            co_occur_obj_correct = pred == co_occur_obj_label
            total_co_occur_obj_correct.append(co_occur_obj_correct.cpu())

        num_correct = meter.cum.reshape(*meter.dims)
        cnt = meter.cnt.reshape(*meter.dims)
        multi_dim_color_acc = num_correct / cnt
        log_dict = {}
        absent_present_str_list = ["absent", "present"]
        absent_present_bg_ratio_list = [1 - args.bg_ratio, args.bg_ratio]
        absent_present_co_occur_obj_ratio_list = [
            1 - args.co_occur_obj_ratio,
            args.co_occur_obj_ratio,
        ]

        weighted_group_acc = 0
        for bg_shortcut in range(len(absent_present_str_list)):
            for second_shortcut in range(len(absent_present_str_list)):
                first_shortcut_mask = (meter.eye_tsr == bg_shortcut).unsqueeze(2)
                co_occur_obj_shortcut_mask = (
                    meter.eye_tsr == second_shortcut
                ).unsqueeze(1)
                mask = first_shortcut_mask * co_occur_obj_shortcut_mask
                acc = multi_dim_color_acc[mask].mean().item()
                bg_shortcut_str = absent_present_str_list[bg_shortcut]
                co_occur_obj_shortcut_str = absent_present_str_list[second_shortcut]
                log_dict[
                    f"{split}_bg_{bg_shortcut_str}"
                    f"_co_occur_obj_{co_occur_obj_shortcut_str}_acc"
                ] = acc
                cur_group_bg_ratio = absent_present_bg_ratio_list[bg_shortcut]
                cur_group_co_occur_obj_ratio = absent_present_co_occur_obj_ratio_list[
                    second_shortcut
                ]
                cur_group_ratio = cur_group_bg_ratio * cur_group_co_occur_obj_ratio
                weighted_group_acc += acc * cur_group_ratio

        bg_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_present_acc"] - weighted_group_acc
        )
        co_occur_obj_gap = (
            log_dict[f"{split}_bg_present_co_occur_obj_absent_acc"] - weighted_group_acc
        )
        both_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_absent_acc"] - weighted_group_acc
        )

        log_dict.update(
            {
                f"{split}_id_acc": weighted_group_acc,
                f"{split}_bg_gap": bg_gap,
                f"{split}_co_occur_obj_gap": co_occur_obj_gap,
                f"{split}_both_gap": both_gap,
            }
        )

        total_bg_correct = torch.cat(total_bg_correct, dim=0)
        total_co_occur_obj_correct = torch.cat(total_co_occur_obj_correct, dim=0)
        total_correct = torch.cat(total_correct, dim=0)

        (
            bg_worst_group_acc,
            co_occur_obj_worst_group_acc,
            both_worst_group_acc,
        ) = meter.get_worst_group_acc()

        log_dict.update(
            {
                f"{split}_bg_worst_group_acc": bg_worst_group_acc,
                f"{split}_co_occur_obj_worst_group_acc": co_occur_obj_worst_group_acc,
                f"{split}_both_worst_group_acc": both_worst_group_acc,
            }
        )

        if args.method == "erm":
            # evaluate cue preference for ERM
            obj_acc = total_correct.float().mean().item()
            bg_acc = total_bg_correct.float().mean().item()
            co_occur_obj_acc = total_co_occur_obj_correct.float().mean().item()

            log_dict.update(
                {
                    f"{split}_cue_obj_acc": obj_acc,
                    f"{split}_cue_bg_acc": bg_acc,
                    f"{split}_cue_co_occur_obj_acc": co_occur_obj_acc,
                }
            )

        self.log_to_wandb(log_dict)

        return log_dict
