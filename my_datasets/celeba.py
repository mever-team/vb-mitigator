import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from my_datasets.utils import (
    TwoCropTransform,
    download_celeba_anno,
    download_celeba_zip,
    get_confusion_matrix,
    get_sampling_weights,
)
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

# from .celeba_torch import CelebA
from torchvision.datasets import CelebA
import os


class BiasedCelebASplit:
    def __init__(self, root, split, transform, target_attr, **kwargs):
        self.transform = transform
        self.target_attr = target_attr
        if os.path.basename(os.path.normpath(root)) == "celeba":
            root = os.path.dirname(root)
        if not os.path.isdir(os.path.join(root, "celeba", "img_align_celeba")):
            download_celeba_zip(root)
        files = [
            "list_eval_partition.txt",
            "list_landmarks_celeba.txt",
            "list_attr_celeba.txt",
            "list_bbox_celeba.txt",
            "list_landmarks_align_celeba.txt",
            "identity_CelebA.txt",
        ]
        flag = False
        for file in files:
            file_path = os.path.join(root, "celeba", file)
            if not os.path.isfile(file_path):
                flag = True
                break
        if flag:
            download_celeba_anno(root)
        self.celeba = CelebA(
            root=root,
            # download=True,
            split="train" if split == "train_valid" else split,
            target_type="attr",
            transform=transform,
        )
        self.bias_idx = 20

        if target_attr == "blonde":
            self.target_idx = 9
            if split in ["train", "train_valid"]:
                save_path = Path(root) / "pickles" / "blonde"
                if save_path.is_dir():
                    print(f"use existing blonde indices from {save_path}")
                    self.indices = pickle.load(open(save_path / "indices.pkl", "rb"))
                else:
                    self.indices = self.build_blonde()
                    print(f"save blonde indices to {save_path}")
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f"indices.pkl", "wb"))
                print(len(self.indices), len(self.celeba.attr))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))

        elif target_attr == "makeup":
            self.target_idx = 18
            self.attr = self.celeba.attr
            self.indices = torch.arange(len(self.celeba))
        else:
            raise AttributeError

        if split in ["train", "train_valid"]:
            save_path = Path(
                os.path.join(root, f"clusters/celeba_rand_indices_{target_attr}.pkl")
            )
            print(save_path.resolve())
            if not save_path.exists():
                rand_indices = torch.randperm(len(self.indices))
                pickle.dump(rand_indices, open(save_path, "wb"))
            else:
                rand_indices = pickle.load(open(save_path, "rb"))

            num_total = len(rand_indices)
            num_train = int(0.8 * num_total)

            if split == "train":
                indices = rand_indices[:num_train]
            elif split == "train_valid":
                indices = rand_indices[num_train:]

            self.indices = self.indices[indices]
            self.attr = self.attr[indices]

        self.targets = self.attr[:, self.target_idx]
        self.biases = self.attr[:, self.bias_idx]

        (
            self.confusion_matrix_org,
            self.confusion_matrix,
            self.confusion_matrix_by,
        ) = get_confusion_matrix(
            num_classes=2, targets=self.targets, biases=self.biases
        )

        print(
            f"Use BiasedCelebASplit \n target_attr: {target_attr} split: {split} \n {self.confusion_matrix_org}"
        )

    def build_blonde(self):
        biases = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(biases == 0) & (targets == 0)]
        non_selects = torch.arange(len(self.celeba))[~((biases == 0) & (targets == 0))]
        np.random.shuffle(selects)
        indices = torch.cat([selects[:2000], non_selects])
        return indices

    def __getitem__(self, index):
        img, _ = self.celeba.__getitem__(self.indices[index])
        target, bias = self.targets[index], self.biases[index]
        # return img, target, bias, index
        return {"inputs": img, "targets": target, "gender": bias, "index": index}

    def __len__(self):
        return len(self.targets)


def get_celeba(
    root,
    batch_size,
    target_attr="blonde",
    split="train",
    num_workers=8,
    aug=False,
    two_crop=False,
    ratio=0,
    img_size=224,
    given_y=True,
    transform=None,
    sampler=None,
):
    logging.info(
        f"get_celeba - split:{split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}"
    )
    if split == "eval":
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        if transform is None:
            if aug:
                transform = T.Compose(
                    [
                        T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                        T.RandomHorizontalFlip(),
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        T.RandomGrayscale(p=0.2),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

            else:
                transform = T.Compose(
                    [
                        T.Resize((img_size, img_size)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

    if two_crop:
        transform = TwoCropTransform(transform)

    dataset = BiasedCelebASplit(
        root=root,
        split=split,
        transform=transform,
        target_attr=target_attr,
    )

    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    if ratio != 0:
        if given_y:
            weights = [
                1 / dataset.confusion_matrix_by[c, b]
                for c, b in zip(dataset.targets, dataset.biases)
            ]
        else:
            weights = [
                1 / dataset.confusion_matrix[b, c]
                for c, b in zip(dataset.targets, dataset.biases)
            ]
        if ratio > 0:
            weights = clip_max_ratio(np.array(weights))
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    # else:
    #     sampler = None
    elif sampler is not None and split == "train":
        if sampler == "weighted":
            # *[torch.tensor(bias) for bias in dataset.bias_targets]
            weights = get_sampling_weights(
                dataset.targets, *[torch.tensor(dataset.biases)]
            )
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=two_crop,
    )
    return dataloader, dataset
