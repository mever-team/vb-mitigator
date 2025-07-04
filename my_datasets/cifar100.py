import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class CustomCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # Fetch the sample and target using the parent class
        sample, target = super().__getitem__(idx)
        # Return the custom dictionary format
        return {"inputs": sample, "targets": target, "unknown": target, "index": idx}


def get_cifar100_loaders(
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
        train_dataset = CustomCIFAR100(
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

        test_dataset = CustomCIFAR100(
            root=root, train=False, download=True, transform=transform
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return test_loader
