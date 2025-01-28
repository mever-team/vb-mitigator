import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json

from datasets.cifar10 import get_cifar10_loaders
from models.resnet_small import resnet20

# Load model
classifier_dict = torch.load(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/cifar10_baselines/dev/mavias/best"
)["model"]

classifier = resnet20(num_classes=10)
classifier.load_state_dict(classifier_dict)
classifier = classifier.to("cuda")


# Load test data
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
test_loader = get_cifar10_loaders(
    root="./data/cifar10",
    split="test",
    transform=transform,
)


class_name_to_idx = {
    "Airplane": 0,
    "Car": 1,
    "Bird": 2,
    "Cat": 3,
    "Deer": 4,
    "Dog": 5,
    "Frog": 6,
    "Horse": 7,
    "Ship": 8,
    "Truck": 9,
}

# Load saved tags (either CSV or JSON file)
with open("tags_above_threshold.json", "r") as json_file:
    tags_above_threshold = json.load(json_file)

# Load test tags from test_tags.csv
test_tags_df = pd.read_csv(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/data/cifar10/test_tags.csv"
)


# Function to check if a sample has any of the saved tags for its class
def has_saved_tag(sample_tags, class_name):
    saved_tags = tags_above_threshold.get(class_name, [])
    # Check if there is any intersection between sample tags and saved tags
    return any(tag in saved_tags for tag in sample_tags)


# Calculate accuracy for samples with saved tags vs those without
classifier.eval()
bs = 128
correct_with_saved_tags = 0
total_with_saved_tags = 0
correct_without_saved_tags = 0
total_without_saved_tags = 0

# Iterate over test data
for idx, batch in enumerate(tqdm(test_loader)):
    img, target = batch["inputs"].to("cuda"), batch["targets"].to("cuda")
    logits, _ = classifier(img)
    pred = logits.argmax(dim=1)

    # Split tags for this batch
    sample_tags_list = [
        row.split(" | ") for row in test_tags_df.iloc[idx : idx + bs]["irrelevant_tags"]
    ]

    for i in range(pred.shape[0]):
        class_name = [
            name for name, idx in class_name_to_idx.items() if idx == target[i].item()
        ][0]
        sample_tags = sample_tags_list[i]

        # Check if sample has any of the saved tags
        has_saved = has_saved_tag(sample_tags, class_name)

        # Update accuracy based on whether it has saved tags or not
        if has_saved:
            correct_with_saved_tags += int(pred[i] == target[i])
            total_with_saved_tags += 1
        else:
            correct_without_saved_tags += int(pred[i] == target[i])
            total_without_saved_tags += 1

# Compute accuracy for both groups
accuracy_with_saved_tags = (
    correct_with_saved_tags / total_with_saved_tags if total_with_saved_tags > 0 else 0
)
accuracy_without_saved_tags = (
    correct_without_saved_tags / total_without_saved_tags
    if total_without_saved_tags > 0
    else 0
)

# Print the results
print(f"Accuracy for samples with saved tags: {accuracy_with_saved_tags * 100:.2f}%")
print(
    f"Accuracy for samples without saved tags: {accuracy_without_saved_tags * 100:.2f}%"
)
