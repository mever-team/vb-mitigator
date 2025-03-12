import csv
import os


import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from datasets.utils import download_waterbirds, get_sampling_weights
import shutil
from torch.utils.data.sampler import WeightedRandomSampler

data_split = {0: "train", 1: "val", 2: "test"}


def fix_data_splits(base_dir, metadata_path):
    """
    Fix the folder structure of the dataset based on metadata.csv.

    Args:
        base_dir (str): The base directory containing the dataset (e.g., 'waterbirds').
        metadata_path (str): Path to the metadata.csv file.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Iterate over metadata rows
    for _, row in metadata.iterrows():
        img_id = row["img_id"]
        img_filename = row["img_filename"].split("/")[-1]
        split = row["split"]
        y = row["y"]  # Class label (0 or 1)

        # Construct expected and current paths
        expected_folder = os.path.join(base_dir, data_split[split], str(y))
        expected_path = os.path.join(expected_folder, img_filename)

        for current_split, split_name in data_split.items():
            current_folder = os.path.join(base_dir, split_name, str(y))
            current_path = os.path.join(current_folder, img_filename)

            if os.path.exists(current_path):
                if current_path != expected_path:
                    # Move file to correct folder
                    os.makedirs(expected_folder, exist_ok=True)
                    shutil.move(current_path, expected_path)
                    # print(f"Moved {img_filename} from {current_folder} to {expected_folder}")


class WaterbirdsDataset(Dataset):
    def __init__(
        self,
        raw_data_path,
        root,
        split="train",
        transform=None,
        target_transform=None,
        return_places=False,
    ) -> None:
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_places = return_places
        self.return_masked = False
        img_data_dir = os.path.join(root, "images", split)

        if not os.path.isdir(os.path.join(root, "images", "test")):
            download_waterbirds(root)
        self.places = {}
        fix_data_splits(
            os.path.join(root, "images"), os.path.join(root, "metadata.csv")
        )
        with open(os.path.join(root, "metadata.csv")) as meta_file:
            csv_reader = csv.reader(meta_file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                img_id, img_filename, y, split_index, place, place_filename = row
                if data_split[int(split_index)] == split:
                    self.places[img_filename.split("/")[-1]] = int(place)
        self.update_data(img_data_dir)

    def update_data(self, data_file_directory):
        self.data_path = []
        self.targets = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-" * 10, f"indexing {self.split} data", "-" * 10)
        for data_class in tqdm(data_classes):
            target = int(data_class)
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class, "*")
            )
            self.data_path += class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, target)
        """
        img_file_path, target = self.data_path[index], self.targets[index]
        img = Image.open(img_file_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(f"target: {target}, background: {p}, index: {index}")
        return {
            "inputs": img,
            "targets": target,
            "background": self.places[img_file_path.split("/")[-1]],
            "index": index,
        }


def get_waterbirds(
    root_dir, batch_size=64, n_workers=4, transform=None, split="train", sampler=None
) -> None:
    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    transform_test = transform
    if transform_test is None:
        transform_test = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0] * scale),
                        int(target_resolution[1] * scale),
                    )
                ),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    transform_train = transform
    if transform_train is None:
        transform_train =   transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # transforms.Compose(
        #     [
        #         transforms.Resize(
        #             (
        #                 int(target_resolution[0] * scale),
        #                 int(target_resolution[1] * scale),
        #             )
        #         ),
        #         transforms.CenterCrop(target_resolution),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ] 
        # )

    if split == "train":
        train_dataset = WaterbirdsDataset(
            raw_data_path=root_dir,
            root=root_dir,
            split="train",
            transform=transform_train,
            return_places=True,
        )
        if sampler == "weighted":
            weights = get_sampling_weights(
                train_dataset.targets,
                *[
                    torch.tensor(
                        [
                            train_dataset.places[img_file_path.split("/")[-1]]
                            for img_file_path in train_dataset.data_path
                        ]
                    )
                ],
            )
            sampler = WeightedRandomSampler(
                weights, len(train_dataset), replacement=True
            )
        else:
            sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True if sampler is None else False,
            num_workers=n_workers,
            sampler=sampler,
        )
        return train_loader, train_dataset
    elif split == "val":
        val_dataset = WaterbirdsDataset(
            raw_data_path=root_dir,
            root=root_dir,
            split="val",
            transform=transform_test,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
        )
        return val_loader, val_dataset
    elif split == "test":
        test_dataset = WaterbirdsDataset(
            raw_data_path=root_dir,
            root=root_dir,
            split="test",
            transform=transform_test,
            return_places=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
        )
        return test_loader, test_dataset
    else:
        raise ValueError(f"split {split} not recognized")
