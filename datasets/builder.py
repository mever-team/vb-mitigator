from .biased_mnist import get_color_mnist


def get_dataset(cfg):
    dataset_name = cfg.DATASET.TYPE
    # method_name = cfg.MITIGATOR.TYPE

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
        # logging.info(
        #     f"confusion_matrix - \n original: {train_loader.dataset.confusion_matrix_org}, \n normalized: {train_loader.dataset.confusion_matrix}"
        # )

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
    return dataset
