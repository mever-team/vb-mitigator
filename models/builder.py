from models import models_dict
from .utils import get_model_dict


def get_model(model_name, num_class):
    model = models_dict[model_name](num_class)
    return model


def get_bcc(cfg, num_class):
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
        else:
            raise ValueError(
                f"Unsupported dataset ({dataset_name}) or bias ({bias_name}) type."
            )
    return models
