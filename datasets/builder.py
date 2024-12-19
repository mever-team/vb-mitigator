from .biased_mnist import get_color_mnist
from .fb_biased_mnist import get_color_mnist as get_fb_color_mnist
from ram import get_transform
from .custom_transforms import get_transform as get_transform_padded


def get_dataset(cfg):
    dataset_name = cfg.DATASET.TYPE
    method_name = cfg.MITIGATOR.TYPE

    if dataset_name == "biased_mnist":
        train_loader = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
            n_confusing_labels=9,
            split="train",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        val_loader = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            split="train_val",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )
        test_loader = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            split="valid",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )
        dataset = {}
        dataset["num_class"] = 10
        dataset["biases"] = ["background"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["root"] = cfg.DATASET.BIASED_MNIST.ROOT
        dataset["target2name"] = {
            0: "number",
            1: "number",
            2: "number",
            3: "number",
            4: "number",
            5: "number",
            6: "number",
            7: "number",
            8: "number",
            9: "number",
        }
        if method_name == "mavias":
            tag_train_loader = get_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader

    elif dataset_name == "fb_biased_mnist":
        train_loader = get_fb_color_mnist(
            cfg.DATASET.FB_BIASED_MNIST.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
            data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
            n_confusing_labels=9,
            split="train",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        val_loader = get_fb_color_mnist(
            cfg.DATASET.FB_BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation1=0.1,
            data_label_correlation2=0.1,
            n_confusing_labels=9,
            split="train_val",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )
        test_loader = get_fb_color_mnist(
            cfg.DATASET.FB_BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation1=0.1,
            data_label_correlation2=0.1,
            n_confusing_labels=9,
            split="valid",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        dataset = {}
        dataset["num_class"] = 10
        dataset["biases"] = ["background", "foreground"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["target2name"] = {
            0: "number",
            1: "number",
            2: "number",
            3: "number",
            4: "number",
            5: "number",
            6: "number",
            7: "number",
            8: "number",
            9: "number",
        }
        dataset["root"] = cfg.DATASET.FB_BIASED_MNIST.ROOT
        if method_name == "mavias":
            tag_train_loader = get_fb_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
    elif dataset_name == "utkface":
        train_loader = get_fb_color_mnist(
            cfg.DATASET.FB_BIASED_MNIST.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
            data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
            n_confusing_labels=9,
            split="train",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        train_loader = get_utk_face(
            cfg.DATASET.UTKFACE.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="train",
            bias_attr=cfg.DATASET.UTKFACE.BIAS,
            image_size=64,
            ratio=cfg.DATASET.UTKFACE.RATIO,
        )

        val_loader = get_utk_face(
            cfg.DATASET.UTKFACE.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="valid",
            bias_attr=cfg.DATASET.UTKFACE.BIAS,
            image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
        )

        test_loader = get_utk_face(
            cfg.DATASET.UTKFACE.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="test",
            bias_attr=cfg.DATASET.UTKFACE.BIAS,
            image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
        )

        dataset = {}
        dataset["num_class"] = 10
        dataset["biases"] = ["background", "foreground"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["target2name"] = {
            0: "number",
            1: "number",
            2: "number",
            3: "number",
            4: "number",
            5: "number",
            6: "number",
            7: "number",
            8: "number",
            9: "number",
        }
        dataset["root"] = cfg.DATASET.FB_BIASED_MNIST.ROOT
        if method_name == "mavias":
            tag_train_loader = get_fb_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader

    return dataset
