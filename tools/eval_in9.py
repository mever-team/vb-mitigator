import argparse
import pandas as pd
import torch.backends.cudnn as cudnn
from configs.cfg import CFG as cfg
from torchvision.models import resnet50
from torch import nn
import torch
from my_datasets.imagenet9 import get_background_challenge_data
from tools.metrics import get_performance
import numpy as np

cudnn.benchmark = True


def val_iter_tags(
    batch, overperforming_tags_df, model, target2name, index_to_tags_test
):
    batch_dict = {}

    inputs = batch["inputs"].to("cuda:0")
    targets = batch["targets"]
    indices = batch["index"]

    tags_dict = {}  # zeros with shape of targets
    tags_dict = {
        f"{row['Class']}_{row['Tag']}": torch.zeros_like(targets)
        for _, row in overperforming_tags_df.iterrows()
    }

    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs, _ = outputs
    batch_dict["predictions"] = torch.argmax(outputs, dim=1)
    batch_dict["targets"] = targets

    for i in range(len(indices)):
        class_name = target2name[targets[i].item()]
        irrelevant_tags = index_to_tags_test[indices[i].item()]
        irrelevant_tags = [tag.strip() for tag in irrelevant_tags.split(", ")]

        for tag in irrelevant_tags:
            if f"{class_name}_{tag}" in tags_dict:
                tags_dict[f"{class_name}_{tag}"][i] = 1

    batch_dict.update(tags_dict)

    return batch_dict


def validate_epoch_tags(
    dataloader, model, overperforming_tags_df, target2name, index_to_tags_test
):
    model.eval()
    with torch.no_grad():
        all_data = {
            f"{row['Class']}_{row['Tag']}": []
            for _, row in overperforming_tags_df.iterrows()
        }

        all_data["targets"] = []
        all_data["predictions"] = []

        for batch in dataloader:
            batch_dict = val_iter_tags(
                batch, overperforming_tags_df, model, target2name, index_to_tags_test
            )
            for key, value in batch_dict.items():
                all_data[key].append(value.detach().cpu().numpy())

        for key in all_data:
            all_data[key] = np.concatenate(all_data[key])
        # metric specific data

        performance = get_performance["wg_ovr_tags"](all_data)
    return performance


def main(cfg, method, seed):
    device = "cuda:0"
    # load model
    if method == "mavias":
        model_root = f"./pretrained/in9/{method}/seed{seed}.pt"
    else:
        model_root = f"./pretrained/in9/{method}/seed{seed}.pth"

    if method == "mavias":
        model = resnet50()
        model.fc = nn.Linear(512 * 4, 9)
        loaded_dict = torch.load(model_root)
        model.load_state_dict(loaded_dict["model"])
    else:
        classifier = nn.Linear(512 * 4, 9)
        model = resnet50()
        model.fc = nn.Identity()
        loaded_dict = torch.load(model_root)
        model.load_state_dict(loaded_dict["backbone"])
        classifier.load_state_dict(loaded_dict["classifier"])
        model.fc = classifier
    model = model.to(device)

    # load data

    target2name = {
        0: "Dog",
        1: "Bird",
        2: "Vehicle",
        3: "Reptile",
        4: "Carnivore",
        5: "Insect",
        6: "Instrument",
        7: "Primate",
        8: "Fish",
    }
    test_loader = get_background_challenge_data(
        root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
        bench="original",  # cfg.DATASET.IMAGENET9.BENCHMARK_TEST,
    )

    # load test tags
    test_tags_df = pd.read_csv(
        "/home/isarridis/projects/vb-mitigator/data/imagenet9/test_tags.csv"
    )
    # load overperforming tags
    overperforming_tags_df = pd.read_csv(
        "/home/isarridis/projects/vb-mitigator/data/imagenet9/overperforming_tags_keep.csv"
    )
    index_to_tags_test = {
        row["index"]: (
            row["irrelevant_tags"].replace(" | ", ", ")
            if isinstance(row["irrelevant_tags"], str)
            else " "
        )
        for _, row in test_tags_df.iterrows()
    }

    performance = validate_epoch_tags(
        test_loader, model, overperforming_tags_df, target2name, index_to_tags_test
    )
    print(performance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("training")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    cfg.freeze()
    main(cfg, args.method, args.seed)
