from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import ram_transform, register_dataset


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # Fetch the sample and target using the parent class
        sample, target = super().__getitem__(idx)
        # Return the custom dictionary format
        return {"inputs": sample, "targets": target, "unknown": target, "index": idx}


def get_cifar10_loaders(
    root, batch_size=128, num_workers=4, transform=None, image_size=32, split="train"
):
    """
    Returns the train, validation, and test loaders for CIFAR-10,
    with `__getitem__` returning batches in the specified format.

    Args:
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: Train, validation, and test data loaders.
    """
    # Define the CIFAR-10 dataset transforms
    if split == "train":
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                    ),  # Standard CIFAR-10 normalization
                ]
            )
        # Load the train and test datasets with the custom dataset class
        train_dataset = CustomCIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return train_loader
    else:
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )

        test_dataset = CustomCIFAR10(
            root=root, train=False, download=True, transform=transform
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return test_loader


@register_dataset("cifar10")
def build_cifar10(cfg):
    dataset_name = cfg.DATASET.TYPE  # noqa: F841 (kept for parity)
    method_name = cfg.MITIGATOR.TYPE
    metric_name = cfg.METRIC
    if method_name == "groupdro":
        raise ValueError(
            "GroupDro requires bias attribute annotations! The cifar10 dataset does not offer such information. Please select another method, or modify cifar10 class so that it incorporates your own bias annotations."
        )
    else:
        train_loader = get_cifar10_loaders(
            root=cfg.DATASET.CIFAR10.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
            split="train",
        )

    val_loader = get_cifar10_loaders(
        root=cfg.DATASET.CIFAR10.ROOT,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
        split="test",
    )

    test_loader = get_cifar10_loaders(
        root=cfg.DATASET.CIFAR10.ROOT,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
        split="test",
    )

    dataset = {}
    dataset["num_class"] = 10
    dataset["num_groups"] = 10
    dataset["biases"] = [cfg.DATASET.CIFAR10.BIAS]
    dataset["dataloaders"] = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    dataset["sets"] = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    dataset["target2name"] = {
        0: "Airplane",
        1: "Car",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck",
    }

    dataset["root"] = cfg.DATASET.CIFAR10.ROOT
    if (
        method_name == "mavias"
        or method_name == "erm_tags"
        or metric_name == "wg_ovr_tags"
    ):
        tag_train_loader = get_cifar10_loaders(
            root=cfg.DATASET.CIFAR10.ROOT,
            batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
            split="train",
            transform=ram_transform(
                image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
            ),
        )
        tag_test_loader = get_cifar10_loaders(
            root=cfg.DATASET.CIFAR10.ROOT,
            batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
            split="test",
            transform=ram_transform(
                image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
            ),
        )

        dataset["dataloaders"]["tag_train"] = tag_train_loader
        dataset["dataloaders"]["tag_test"] = tag_test_loader
    return dataset
