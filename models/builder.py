"""
This module provides functions to retrieve and construct models for various datasets and biases.

Functions:
    get_model(model_name, num_class, pretrained=False):

    get_bcc(cfg, num_class):
"""

from models.resnet import set_resnet_fc
from models import models_dict
from .utils import get_model_dict


def get_model(model_name, num_class, pretrained=False):
    """
    Retrieve a model from the models dictionary.

    Args:
        model_name (str): The name of the model to retrieve.
        num_class (int): The number of classes for the model's output layer.
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Defaults to False.

    Returns:
        torch.nn.Module: The requested model initialized with the specified number of classes and pre-trained weights if specified.
    """
    model = models_dict[model_name](num_class, pretrained)
    return model


def get_bcc(cfg, num_class):
    """
    Constructs and returns a dictionary of bias capturing models based on the provided configuration and number of classes.

    Args:
        cfg (object): Configuration object containing dataset and bias information.
        num_class (int): Number of classes for the classification task.

    Returns:
        dict: A dictionary where keys are bias names and values are the corresponding models.

    Raises:
        ValueError: If the dataset or bias type is unsupported.

    The function supports the following dataset and bias combinations:
        - biased_mnist: color
        - fb_biased_mnist: fgcolor, bgcolor
        - utkface: race
        - waterbirds: background
        - celeba: gender
    """
    dataset_name = cfg.DATASET.TYPE

    bias_names = cfg.DATASET.BIASES

    models = {}
    for bias_name in bias_names:
        model_dict = get_model_dict(dataset_name, bias_name)
        if (dataset_name == "biased_mnist") and (bias_name == "color"):
            model = models_dict["simple_conv"](num_class)
            model.load_state_dict(model_dict["model"])
            models[bias_name] = model
        elif (dataset_name == "fb_biased_mnist") and (
            (bias_name == "fgcolor") or (bias_name == "bgcolor")
        ):
            model = models_dict["simple_conv"](num_class)
            model.load_state_dict(model_dict["model"])
            models[bias_name] = model
        elif (dataset_name == "utkface") and (bias_name == "race"):
            model = models_dict["resnet18"](num_class)
            model.load_state_dict(model_dict["model"])
            models[bias_name] = model
        elif (dataset_name == "waterbirds") and (bias_name == "background"):
            model = set_resnet_fc(models_dict["resnet50_def"](), num_class)
            model.load_state_dict(model_dict)
            models[bias_name] = model
        elif dataset_name == "celeba" and (bias_name == "gender"):
            model = models_dict["resnet18"](num_class)
            model.load_state_dict(model_dict["model"])
            models[bias_name] = model
        elif dataset_name == "urbancars" and (bias_name == "background"):
            model = models_dict["resnet50"](num_class)
            model.load_state_dict(model_dict["model"])
            models[bias_name] = model
        elif dataset_name == "urbancars" and (bias_name == "object"):
            model = models_dict["resnet50"](num_class)
            model.load_state_dict(model_dict["model"])
            models[bias_name] = model
        else:
            raise ValueError(
                f"Unsupported dataset ({dataset_name}) or bias ({bias_name}) type."
            )
    return models
