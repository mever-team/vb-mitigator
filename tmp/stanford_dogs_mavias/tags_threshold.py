import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json

from datasets.stanford_dogs import get_stanford_dogs_loader
from models.resnet import ResNet18

# Load model
classifier_dict = torch.load(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/stanford_dogs_baselines/dev/erm/best"
)["model"]

classifier = ResNet18(num_classes=120)
classifier.load_state_dict(classifier_dict)
classifier = classifier.to("cuda")


test_loader = get_stanford_dogs_loader(
    root="./data/stanford-dogs-dataset",
    split="test",
)

class_names = test_loader.dataset.classes
class_names = [name + " dog" for name in class_names]
class_name_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Dictionary to hold tags for each class
class_tags = {}
bs = 128
# Load relevant tags from CSVs
for class_name, class_idx in class_name_to_idx.items():
    file_name = f"/mnt/cephfs/home/gsarridis/projects/vb-mitigator/data/stanford-dogs-dataset/relevant_tags/llama3_bs100_{class_idx}_{class_name}.csv"
    try:
        # Load CSV file into a DataFrame
        df = pd.read_csv(file_name)

        # Store tags in a dictionary (as a list)
        class_tags[class_name] = df["tags"].tolist()
        print(f"Loaded {len(class_tags[class_name])} tags for class '{class_name}'")
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except pd.errors.EmptyDataError:
        print(f"File {file_name} is empty.")

# Load test tags from test_tags.csv
test_tags_df = pd.read_csv(
    "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/data/stanford-dogs-dataset/test_tags.csv"
)


def get_irrelevant_tags(sample_tags, relevant_tags):
    # Return the list of irrelevant tags by checking if each sample tag is not in relevant tags
    return [tag for tag in sample_tags if tag not in relevant_tags]


# Calculate accuracy based on exact irrelevant tags for each class
classifier.eval()

# Dictionary to hold the count of correct predictions for each irrelevant tag within each class
class_tag_correct_counts = {class_name: {} for class_name in class_name_to_idx}
class_tag_total_counts = {class_name: {} for class_name in class_name_to_idx}

correct = 0
total = len(test_loader.dataset)

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
        relevant_tags = class_tags.get(class_name, [])

        # Get the list of irrelevant tags
        irrelevant_tags = get_irrelevant_tags(sample_tags, relevant_tags)

        cor = int(pred[i] == target[i])  # Check if prediction is correct

        # Update accuracy for each irrelevant tag within the current class
        for tag in irrelevant_tags:
            if tag not in class_tag_correct_counts[class_name]:
                class_tag_correct_counts[class_name][tag] = cor
                class_tag_total_counts[class_name][tag] = 1
            else:
                class_tag_correct_counts[class_name][tag] += cor
                class_tag_total_counts[class_name][tag] += 1

# Now find and store the tags with accuracy > 77% for each class
threshold_accuracy = 0.7719  # 77% accuracy threshold

# Dictionary to store tags with accuracy > 77% for each class
tags_above_threshold = {class_name: [] for class_name in class_name_to_idx}
tags_above_threshold_acc = {class_name: [] for class_name in class_name_to_idx}

# Check accuracy for each tag and store if it is above the threshold
for class_name in class_tag_correct_counts:
    for tag in class_tag_correct_counts[class_name]:
        accuracy = (
            class_tag_correct_counts[class_name][tag]
            / class_tag_total_counts[class_name][tag]
        )
        if accuracy > threshold_accuracy:
            tags_above_threshold[class_name].append(tag)
            tags_above_threshold_acc[class_name].append(accuracy)

# Save tags with accuracy > 77% to a file

# Option 1: Save as a CSV file
# Create a DataFrame and write it to CSV
tags_for_csv = []
for class_name, tags, accs in zip(
    tags_above_threshold.keys(),
    tags_above_threshold.values(),
    tags_above_threshold_acc.values(),
):
    for tag, acc in zip(tags, accs):
        tags_for_csv.append([class_name, tag, acc])

tags_df = pd.DataFrame(tags_for_csv, columns=["Class", "Tag", "Acc"])
tags_df.to_csv("tags_above_threshold.csv", index=False)

print("\nTags with accuracy > 77% saved to 'tags_above_threshold.csv'.")

# Option 2: Save as a JSON file
with open("tags_above_threshold.json", "w") as json_file:
    json.dump(tags_above_threshold, json_file)

print("\nTags with accuracy > 77% saved to 'tags_above_threshold.json'.")
