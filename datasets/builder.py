from datasets.celeba import get_celeba
from datasets.cifar100 import get_cifar100_loaders
from datasets.imagenet9 import get_background_challenge_data, get_imagenet9l
from .biased_mnist import get_color_mnist
from .fb_biased_mnist import get_color_mnist as get_fb_color_mnist
from ram import get_transform
from .custom_transforms import get_transform as get_transform_padded
from .utk_face import get_utk_face
from .waterbirds import get_waterbirds
from .cifar10 import get_cifar10_loaders


def get_dataset(cfg):
    dataset_name = cfg.DATASET.TYPE
    method_name = cfg.MITIGATOR.TYPE

    if dataset_name == "biased_mnist":
        if method_name == "groupdro":
            train_loader = get_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                sampler="weighted",
            )
        else:
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
        dataset["num_groups"] = 10 * 10
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
        if method_name == "groupdro":
            train_loader = get_fb_color_mnist(
                cfg.DATASET.FB_BIASED_MNIST.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                sampler="weighted",
            )
        else:
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
        dataset["num_groups"] = 10 * 10 * 10
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
        if method_name == "groupdro":
            train_loader = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
                ratio=cfg.DATASET.UTKFACE.RATIO,
                sampler="weighted",
            )
        else:
            train_loader = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
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
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = [cfg.DATASET.UTKFACE.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["target2name"] = {0: "male", 1: "female"}
        dataset["root"] = cfg.DATASET.UTKFACE.ROOT
        dataset["ba_groups"] = cfg.DATASET.UTKFACE.BIAS_ALIGNED
        # print(dataset["ba_groups"])
        if method_name == "mavias":
            tag_train_loader = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                split="train",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                ratio=cfg.DATASET.UTKFACE.RATIO,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
    elif dataset_name == "waterbirds":
        if method_name == "groupdro":
            train_loader = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                split="train",
                sampler="weighted",
            )
        else:
            train_loader = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                split="train",
            )

        val_loader = get_waterbirds(
            cfg.DATASET.WATERBIRDS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            n_workers=cfg.DATASET.NUM_WORKERS,
            split="val",
        )
        test_loader = get_waterbirds(
            cfg.DATASET.WATERBIRDS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            n_workers=cfg.DATASET.NUM_WORKERS,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 2
        dataset["biases"] = ["background"]
        dataset["num_groups"] = 2 * 2
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["root"] = cfg.DATASET.WATERBIRDS.ROOT
        dataset["target2name"] = {
            0: "bird",
            1: "bird",
        }
        if method_name == "mavias":
            tag_train_loader, _, _ = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
    elif dataset_name == "celeba":
        if method_name == "groupdro":
            train_loader = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
                ratio=cfg.DATASET.CELEBA.RATIO,
                sampler="weighted",
            )
        else:
            train_loader = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
                ratio=cfg.DATASET.CELEBA.RATIO,
            )

        val_loader = get_celeba(
            cfg.DATASET.CELEBA.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="valid",
            target_attr=cfg.DATASET.CELEBA.TARGET,
            img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
        )

        test_loader = get_celeba(
            cfg.DATASET.CELEBA.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="test",
            target_attr=cfg.DATASET.CELEBA.TARGET,
            img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
        )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = [cfg.DATASET.CELEBA.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        if cfg.DATASET.CELEBA.TARGET == "blonde":
            dataset["target2name"] = {0: "non blonde", 1: "blonde"}
        elif cfg.DATASET.CELEBA.TARGET == "makeup":
            dataset["target2name"] = {0: "no makeup", 1: "makeup"}
        else:
            raise ValueError("Target attribute should be either blonde or makeup.")
        dataset["root"] = cfg.DATASET.CELEBA.ROOT
        dataset["ba_groups"] = cfg.DATASET.CELEBA.BIAS_ALIGNED
        # print(dataset["ba_groups"])
        if method_name == "mavias":
            tag_train_loader = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                split="train",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                ratio=cfg.DATASET.CELEBA.RATIO,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
    elif dataset_name == "imagenet9":
        if method_name == "groupdro":
            raise ValueError(
                "GroupDro requires bias attribute annotations! The imagenet dataset does not offer such information. Please select another method, or modify imagenet class so that it incorporates your own bias annotations."
            )
        else:
            train_loader = get_imagenet9l(
                root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            )

        val_loader = get_background_challenge_data(
            root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            bench=cfg.DATASET.IMAGENET9.BENCHMARK_VAL,
        )

        test_loader = get_background_challenge_data(
            root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            bench=cfg.DATASET.IMAGENET9.BENCHMARK_TEST,
        )

        dataset = {}
        dataset["num_class"] = 9
        dataset["num_groups"] = 9
        dataset["biases"] = [cfg.DATASET.IMAGENET9.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        dataset["target2name"] = {
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

        dataset["root"] = cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG
        if method_name == "mavias":
            tag_train_loader = get_imagenet9l(
                root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
    elif dataset_name == "cifar10":
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
        if method_name == "mavias":
            tag_train_loader = get_cifar10_loaders(
                root=cfg.DATASET.CIFAR10.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
                split="train",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader = get_cifar10_loaders(
                root=cfg.DATASET.CIFAR10.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
                split="test",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )

            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "cifar100":
        if method_name == "groupdro":
            raise ValueError(
                "GroupDro requires bias attribute annotations! The cifar100 dataset does not offer such information. Please select another method, or modify cifar100 class so that it incorporates your own bias annotations."
            )
        else:
            train_loader = get_cifar100_loaders(
                root=cfg.DATASET.CIFAR100.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="train",
            )

        val_loader = get_cifar100_loaders(
            root=cfg.DATASET.CIFAR100.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
            split="test",
        )

        test_loader = get_cifar100_loaders(
            root=cfg.DATASET.CIFAR100.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 10
        dataset["num_groups"] = 10
        dataset["biases"] = [cfg.DATASET.CIFAR100.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        class_names = (
            train_loader.dataset.classes
        )  # This returns a list of class names in order of indices

        # Create a dictionary mapping index to class name
        dataset["target2name"] = {idx: name for idx, name in enumerate(class_names)}

        dataset["root"] = cfg.DATASET.CIFAR100.ROOT
        if method_name == "mavias":
            tag_train_loader = get_cifar100_loaders(
                root=cfg.DATASET.CIFAR100.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="train",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader = get_cifar100_loaders(
                root=cfg.DATASET.CIFAR100.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="test",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )

            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader

    return dataset
